# Declaração de uso de IA generativa (Seção 8 do enunciado)

Atualizar a cada uso relevante; copiar a versão final para o relatório.

| Data | Ferramenta | Finalidade | Revisado por |
|---|---|---|---|
| 2026-10-21 | Claude Code (Anthropic) | Organização do ambiente: estrutura de pastas, ambiente virtual, utilitários de carga/pipeline/auditoria de vazamento e guias de trabalho | |
| 2026-10-21 | Claude Code (Anthropic) | Etapa 1: rascunho do notebook de análise exploratória (`notebooks/src/01_eda.py`) — código das análises, gráficos e texto de interpretação, a ser revisado pela equipe | |
| 2026-10-31 | Claude Code (Anthropic) | Adaptação do ambiente à base atualizada: diagnóstico da base (vazamento, ruído de rótulo, ausentes), reescrita dos utilitários do pacote, testes, guias e notebook 01; experimentos de referência em `tools/rodar_experimentos.py`. Tudo a ser revisado e validado pela equipe | |
| 2026-11-01 | Claude Code (Anthropic) | Versão resumida da Etapa 1 em PDF (`results/01_eda_resumo.pdf`): condensação do texto do notebook e tutorial de reprodução no Google Colab, a ser revisado pela equipe | |
| 2026-11-01 | Claude Code (Anthropic) | Versão LaTeX da Etapa 1 para o Overleaf (`reports/overleaf/`): reescrita do texto em linguagem mais natural e tutorial do pacote local no início, a ser revisado pela equipe | |
| 2026-11-07 | Claude Code (Anthropic) | Adequação ao enunciado atualizado: comparação das versões, nome `submissao_<N>.csv` e validação do formato da Seção 3.1 (`write_submission`/`check_submission`), precisão e recall por classe nas métricas de CV, testes e atualização dos guias, a ser revisado pela equipe | |
