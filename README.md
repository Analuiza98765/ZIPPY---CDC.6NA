# Zippy — Simulador de Robô Móvel

Simulador de navegação de robô móvel em uma planta residencial, com física diferencial,
LiDAR de 360°, odometria, grade de ocupação, planejamento A* e replanejamento.

> **Escopo:** este projeto é uma simulação de software. O LiDAR, a odometria, os motores,
> a posição e os obstáculos são simulados; não há conexão com um robô físico nem painel
> recebendo dados de hardware. Os parâmetros físicos são estimativas e não devem ser
> usados para controlar um robô real sem validação e calibração.

## Requisitos

- Python 3.10 ou superior
- pygame 2.5 ou superior
- numpy 1.24 ou superior

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

## Executar

Mantenha estes arquivos na mesma pasta:

- `Projeto_Zippy.py`
- `Planta_Baixa_Modern_House.jpeg`
- `requirements.txt`

O arquivo `Old.Projeto_Zippy.py` é uma cópia antiga e não é necessário para executar
a versão atual.

Inicie com:

```bash
python Projeto_Zippy.py
```

## Navegação simulada e segurança de movimento

Os itens abaixo descrevem proteções geométricas e de controle dentro da simulação;
eles não são controles de cibersegurança nem foram validados em hardware físico.

- O planejador A* trabalha em uma grade de ocupação inflada pelo raio do robô
  e por uma margem de segurança.
- A suavização do caminho verifica a linha com amostragem densa e rejeita
  diagonais que cortem quinas de células ocupadas.
- O controlador reduz a velocidade e o avanço de mira para contornar curvas
  com menor risco de cortar cantos.
- A física verifica o volume varrido pelo robô durante cada passo, impedindo
  que atravesse paredes e obstáculos.
- Os vãos das portas fazem parte da geometria navegável, mas não são desenhados
  como linhas brancas na interface.
- Passagens sob móveis são liberadas somente para móveis marcados como
  transitáveis e cuja altura livre estimada seja maior que a altura do robô
  mais a margem vertical. Apoios continuam sendo obstáculos.

## Ajustes físicos

A planta é uma imagem superior e não informa alturas reais. Os parâmetros
`ROBOT_HEIGHT`, `HEIGHT_MARGIN` e os valores de `MOVEIS_PASSAVEIS` são
estimativas configuráveis. Antes de usar em um robô físico, meça o equipamento,
as alturas dos móveis, as larguras das portas e calibre a escala da planta.

A margem de navegação é definida por `INFLATE_CELLS`; aumentá-la torna o trajeto
mais conservador, mas pode impedir a passagem por corredores estreitos.

## Controles

Os controles e atalhos disponíveis são apresentados na barra inferior e no
painel lateral do simulador.

## Validação de segurança

Os testes controlados dos caminhos de cenário e save podem ser repetidos com:

```bash
python -m unittest discover -s tests -v
```

O teste intercepta `eval()` e `pickle.load()` com stubs; não executa payloads
nem desserializa arquivos pickle. Portanto, é uma verificação dinâmica limitada
de comportamento e alcance dos caminhos, não um pentest completo. O arquivo
`Projeto_Zippy_vulneravel.py` contém exemplos intencionalmente inseguros e não
deve ser executado com arquivos de origem desconhecida. Consulte
`Documentacao-Zippy-NEXUS-ExpoTech-2026-Entrega-Revisada.pdf` para o relatório, os resultados e
as limitações da validação.
