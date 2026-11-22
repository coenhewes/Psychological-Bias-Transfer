#!/bin/bash
cd /home/forge/Psychological-Bias-Transfer/results
echo "# CLEAN RUN LOG - PBT PIPELINE" > CLEAN_RUN_LOG.md
echo "Generated: $(date)" >> CLEAN_RUN_LOG.md
echo "" >> CLEAN_RUN_LOG.md

echo "## Paired T-Tests (Bonferroni-Corrected)" >> CLEAN_RUN_LOG.md
echo "\`\`\`csv" >> CLEAN_RUN_LOG.md
cat paired_ttests.csv >> CLEAN_RUN_LOG.md
echo "\`\`\`" >> CLEAN_RUN_LOG.md
echo "" >> CLEAN_RUN_LOG.md

echo "## ANOVA - Rumination" >> CLEAN_RUN_LOG.md
echo "\`\`\`csv" >> CLEAN_RUN_LOG.md
cat anova_rumination.csv >> CLEAN_RUN_LOG.md
echo "\`\`\`" >> CLEAN_RUN_LOG.md
echo "" >> CLEAN_RUN_LOG.md

echo "## Power Analysis" >> CLEAN_RUN_LOG.md
echo "\`\`\`csv" >> CLEAN_RUN_LOG.md
cat power_analysis.csv >> CLEAN_RUN_LOG.md
echo "\`\`\`" >> CLEAN_RUN_LOG.md

echo "## Seed Variance Check" >> CLEAN_RUN_LOG.md
echo "\`\`\`csv" >> CLEAN_RUN_LOG.md
cat seed_variance_check.csv >> CLEAN_RUN_LOG.md
echo "\`\`\`" >> CLEAN_RUN_LOG.md
