# Instruções

Instalar Python Manager

Instalar python 3.12 e criar environment

```shell
py install 3.12
py -3.12 -m venv venv
```

## Instalar dependências do script de transcrição

```shell

.\venv\Scripts\python.exe -m pip install --no-deps --index-url https://download.pytorch.org/whl/cu128 "torch==2.8.0+cu128" "torchaudio==2.8.0+cu128" "torchvision==0.23.0+cu128"

.\venv\Scripts\python.exe -m pip install whisperx requests

.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Transcrever e revisar

O script `transcribe.py` gera um arquivo `.txt` com o mesmo nome-base do áudio,
contendo a transcrição e os oradores identificados. Por exemplo,
`reuniao.mp3` gera `reuniao.txt`:

```shell
.\venv\Scripts\python.exe transcribe.py caminho\para\reuniao.mp3 pasta_destino
```

Para gerar o resumo com o Ollama, execute `review_transcript.py` informando o
arquivo de transcrição. Ao informar somente o nome do arquivo, o script procura
primeiro em `transcripts/`. Sem um caminho de saída, o resumo é salvo em
`output/`, com o mesmo nome-base e o sufixo `_summary` (por exemplo,
`output/reuniao_summary.txt`). O Ollama deve estar em execução com o modelo
`gemma4:12b` disponível. Os arquivos diretamente presentes em `prompts/` são
executados em ordem alfabética: o primeiro recebe a transcrição e cada
subsequente recebe a resposta da etapa anterior. Cada prompt deve conter
`{transcript}`, substituído pelo texto recebido nessa etapa. Se não houver
arquivos em `prompts/`, usa-se `default_review_prompt.md`, ao lado de
`review_transcript.py`.

As respostas intermediárias são gravadas em `temp/`, relativo ao diretório de
execução, com o número da etapa e o nome do arquivo de prompt (por exemplo,
`temp/01_primeiro.md.txt`). Esses arquivos são sobrescritos em execuções
posteriores; arquivos de etapas anteriores que não forem reutilizados não são
apagados. A última resposta é gravada no arquivo de saída final. A configuração
do modelo é carregada de `review_model.json`, também ao lado de
`review_transcript.py`.

As tags de configuração ficam no início do conteúdo do arquivo de prompt,
antes das instruções. É possível combinar várias tags por linha ou distribuí-las
em várias linhas; linhas em branco também são permitidas no cabeçalho:

```text
[clear] [no-think]

Instruções do prompt...
{transcript}
```

Cada linha de tags deve conter somente tags. O cabeçalho é removido antes de
enviar o prompt ao Ollama. Tags desconhecidas ou repetidas e tags conhecidas
misturadas com texto são rejeitadas. Tags deixadas no nome do arquivo também
são rejeitadas: mova-as para o cabeçalho e renomeie o arquivo. A ordem de
execução continua sendo determinada pelo nome completo dos arquivos.

Edite `review_model.json` para trocar o modelo ou as opções enviadas ao Ollama.
As chaves de `options` são mescladas aos valores padrão definidos no script:
as opções do JSON substituem valores padrão com a mesma chave, e as omitidas
mantêm o padrão.
A temperatura padrão é `0`, para reduzir a aleatoriedade na revisão. Isso e as
instruções de fidelidade do prompt não garantem que o resumo esteja livre de
inferências; confira informações importantes na transcrição.
Use as chaves `stream` (booleano) e `think` (booleano, texto ou `null`) na raiz
do JSON para configurar o streaming e o thinking. Com `stream: true`, o script
exibe os trechos conforme chegam; com `stream: false`, aguarda a resposta
completa. Cada campo omitido entre `stream` e `think` é tratado como `false`.
Para desativar o thinking apenas em uma etapa, inclua a tag literal
`[no-think]` no cabeçalho do prompt. Essa etapa envia `think: false`,
independentemente da configuração global; as demais continuam usando o valor
de `review_model.json`.

Um prompt com a tag literal `[replace]` no cabeçalho deve retornar apenas uma lista
JSON, por exemplo `[{"old": "termo antigo", "new": "termo novo"}]`.
O Ollama recebe também um esquema JSON para orientar o formato da resposta.
O script substitui todas as ocorrências exatas de cada `old` no texto recebido
pela etapa: a transcrição original na primeira etapa ou o resultado da etapa
anterior nas demais. O arquivo de transcrição não é alterado por essas
substituições. Uma lista vazia (`[]`) mantém o texto intacto. As substituições são
simultâneas, sem reaplicar outras regras ao texto inserido. Em ocorrências
sobrepostas, prevalece a primeira posição e, nessa posição, o `old` mais longo.
Valores `old` vazios, repetidos, ausentes ou totalmente encobertos por outra
regra são rejeitados, assim como JSON inválido ou campos incorretos. O processo
para em caso de erro, sem enviar o resultado à próxima etapa.
A lista retornada é salva em `temp/` com o sufixo `.replacements.json`, mesmo
quando a lista for inválida, para facilitar a conferência. O arquivo
intermediário `.txt` contém o texto após as substituições.

Um prompt com a tag `[clear]` recebe o texto da etapa com as linhas numeradas
a partir de 1, cada número seguido por uma tabulação. O prompt deve retornar
somente uma lista JSON de números de linhas a remover, por exemplo `[2, 5]`,
ou `[]` para não remover nenhuma. Os números se referem à entrada original
daquela etapa, independentemente da ordem na lista. O script preserva o texto
e as quebras de linha das linhas mantidas e passa o resultado à próxima etapa.
O total de linhas é informado ao modelo e limita o esquema JSON. Durante o
recebimento, números repetidos, não inteiros ou fora do intervalo e JSON
inválido interrompem a geração, sem aplicar uma limpeza parcial. Para detectar
erros durante a geração, etapas `[clear]` usam streaming internamente mesmo
quando `stream` é `false`; nesse caso, o resultado só é exibido ao terminar.
A resposta é salva em `temp/` com o sufixo `.clear.json`, inclusive a parte
recebida até uma falha de validação; o arquivo intermediário `.txt` contém o
texto filtrado.
As tags `[clear]` e `[replace]` não podem ser usadas no mesmo prompt, mas
ambas podem ser combinadas com `[no-think]`.

O script envia a configuração em cada etapa, acrescentando o prompt; o
thinking é exibido quando retornado pelo modelo, e o arquivo de saída contém
somente o resultado final.
Se o Ollama encerrar por limite de geração ou de contexto (`done_reason:
length`), o script interrompe o pipeline em vez de aceitar uma resposta
possivelmente incompleta. Nesse caso, reduza a entrada ou confira as opções
`num_ctx` e `num_predict` em `review_model.json`.

```shell
.\venv\Scripts\python.exe review_transcript.py reuniao.txt
```

Também é possível usar a revisão em qualquer outra transcrição e informar
explicitamente o caminho do arquivo de saída. Se o segundo caminho for um
diretório existente, o resumo será salvo nele usando o nome da transcrição com
o sufixo `_summary`:

```shell
.\venv\Scripts\python.exe review_transcript.py outra_transcricao.txt meu_resumo.txt
.\venv\Scripts\python.exe review_transcript.py outra_transcricao.txt .\
```