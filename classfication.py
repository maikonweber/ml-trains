from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachoro1 = [0, 1, 1]
cachoro2 = [1, 0, 1]
cachoro3 = [1, 1, 1]

dados = [porco1, porco2, porco3, cachoro1, cachoro2, cachoro3]

classes = [1, 1, 1, 0, 0, 0]

model = LinearSVC()

model.fit(dados, classes)


misterioso1 = [1,1,1]
misterioso2 = [1, 1, 0]
misterioso3 = [0, 1, 1]

teste = [misterioso1, misterioso2, misterioso3]

previsoes = model.predict(teste)

teste_classes = [0, 1, 1]

corretos = (previsoes == teste_classes).sum()

total = len(teste)

taxa = corretos/total

taxa_acerto = accuracy_score(teste_classes, previsoes)

print("taxa de acerto", taxa_acerto * 100)
