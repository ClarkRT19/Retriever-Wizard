Dansk (opsummering)

Jeg haft brug for et værktøj, der lader mig undersøge og sammenligne enkelte værker/anskuelsestavler op imod hele korpus'er (anskuelsestavler, kunstværker eller kombinationer) og samtidig danne mig et overblik via projektioner. Tidligere brugte jeg PixPlot, Collection Space Navigator og Orange Data Mining; de er brugbare og visuelt "flotte", men de egner sig ikke til den form for kombination af nær- og fjernanalyse i samme arbejdsgang. 
Derfor byggede jeg Retriever Wizard, som muliggør metoden: Mixed Viewing.

Med mixed viewing kombinere jeg klassisk kunsthistorisk værkanalyse med distant viewing metodologi (Tilton and Taylor, Moretti)
Meget lavpraktisk anvender jeg følgende pipeline:
1. Computer Vision behandling; billedkorpus bliver bearbejdet af en model, fx CLIP eller sigLIP2, resultatet bliver embeddings.

2. Embeddings projekteres og kvalificeres; forskellige tilgange; gennemgang af cluster ligesom i Ai-nskuelse, og/eller mere nær gennemgang ud fra specifikke cases. resultatet er en bedømmelse af modellens output i relation til humanistiske analyser. (fx hvilke spor af bias ser vi? Har det en reel betydning? Hvordan vurderer modellen den type værker vi er in-teresserede i baseret på naboskaber og distance.)  

3. Analyse af værkerne og deres relationer. Ved dette punkt bliver det nu muligt at sige noget om værkerne i sig selv og deres relationer i datasættet. 
