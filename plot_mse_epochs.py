import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

# Encontra todos os arquivos de histórico de épocas
files = glob.glob('history_epoch_k*.csv')

plt.figure(figsize=(10, 6))

for file in files:
    # Extrai o valor de k do nome do arquivo
    match = re.search(r'k(\d+)', file)
    k = match.group(1) if match else file

    # Lê o CSV
    df = pd.read_csv(file)
    epochs = df['epoch']
    mse = df['mse']

    # Última época (ponto de parada)
    stopped_epoch = epochs.iloc[-1]

    # Plota a curva
    plt.plot(epochs, mse, label=f'k={k} (stop@{stopped_epoch})')

    # Marca o ponto de parada apenas com o ponto, sem texto
    plt.scatter(stopped_epoch, mse.iloc[-1], marker='o', color=plt.gca().lines[-1].get_color())

plt.xlabel('Época')
plt.ylabel('MSE')
plt.title('MSE por Época para diferentes valores de k')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('mse_por_epoca.png', dpi=150)
plt.show()
