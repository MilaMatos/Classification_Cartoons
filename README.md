# Classification Cartoons

### O objetivo do algoritmo é informar qual o desenho mais provável de ser com base no dataset disponível

-> Atualmente na saída da classificação da imagem é dado as 3 maiores probabilidades de qualquer desenho animado ser a da imagem dada, além de dar as probabilidades dentro de cada classe de desenho.

- Estão sendo usados 800 imagens de cada desenho abaixo:

1. Apenas_Um_Show
2. Bob_Esponja
3. Bungo_Stray_Dogs
4. Kick_Buttowski
5. Looney_Tunes
6. Madeline
7. Padrinhos_Magicos
8. Pica_Pau


* Para validação está sendo feito um K-fold com 5 folds e sendo rodado em 5 épocas. Sendo gravado também a acurácia de treino e validação para cada época e fold e a matriz de confusão e precisão de cada classe para cada fold feito.