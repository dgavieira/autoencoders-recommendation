#!/usr/bin/env python3
# autoencoder_gpu_full_progress.py

import csv
import cupy as cp
import numpy as np
import random
from time import time
from tqdm import tqdm

ALPHA = 0.02
LAMBDA = 0.0002
KS = [50, 75, 100]
STOP_MSE = 0.1
MAX_EPOCHS = 500
MIN_IMPROVEMENT = 5e-5
BATCH_SIZE = 1024
SEED = 42
cp.random.seed(SEED)
random.seed(SEED)

# ---------- Load data ----------
def load_movies(movies_csv_path):
    movie_titles = {}
    with open(movies_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row['movieId'])
            movie_titles[mid] = row.get('title', str(mid))
    return movie_titles

def load_ratings(ratings_csv_path):
    ratings = []
    users = set()
    movies = set()
    with open(ratings_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                u = int(row['userId'])
                m = int(row['movieId'])
                r = float(row['rating'])
            except Exception:
                continue
            ratings.append((u, m, r))
            users.add(u)
            movies.add(m)
    return ratings, sorted(users), sorted(movies)

def build_matrices(ratings, users_list, movies_list):
    user_to_idx = {u:i for i,u in enumerate(users_list)}
    movie_to_idx = {m:i for i,m in enumerate(movies_list)}
    U = len(users_list)
    N = len(movies_list)
    R = cp.zeros((U, N), dtype=cp.float32)
    M = cp.zeros((U, N), dtype=bool)
    for (u, m, r) in ratings:
        if m in movie_to_idx and u in user_to_idx:
            ui = user_to_idx[u]
            mi = movie_to_idx[m]
            R[ui, mi] = r
            M[ui, mi] = True
    return R, M, user_to_idx, movie_to_idx

def initialize_weights_xavier(k, n):
    """Inicializa W e V usando Xavier uniform"""
    limit = cp.sqrt(6 / (k + n))
    W = cp.random.uniform(-limit, limit, size=(k, n), dtype=cp.float32)
    V = cp.random.uniform(-limit, limit, size=(n, k), dtype=cp.float32)
    return W, V


def train_autoencoder_batches(R, M, k, alpha=ALPHA, lam=LAMBDA,
                              stop_mse=STOP_MSE, max_epochs=MAX_EPOCHS,
                              min_improvement=MIN_IMPROVEMENT,
                              batch_size=BATCH_SIZE):
    U, N = R.shape
    W, V = initialize_weights_xavier(k, N)  # Xavier init
    R_mean = cp.mean(R, axis=1, keepdims=True)
    S = [(u, i) for u in range(U) for i in range(N) if M[u, i]]
    history = []
    epoch_mse_history = []
    start_time = time()
    last_mse = float('inf')

    # Learning rate adaptativo
    decay = 0.9
    patience = 5
    wait = 0

    # Normaliza uma vez
    R_norm = (R - R_mean) / 4.5  

    for epoch in range(1, max_epochs+1):
        random.shuffle(S)
        batch_count = (len(S) + batch_size - 1) // batch_size

        with tqdm(range(batch_count), desc=f"Epoch {epoch:03d}", leave=False) as pbar:
            for b in pbar:
                batch = S[b*batch_size:(b+1)*batch_size]
                if not batch:
                    continue

                u_idx = cp.array([x[0] for x in batch])
                i_idx = cp.array([x[1] for x in batch])

                r_u_batch = R_norm[u_idx, :]          
                V_batch = V[i_idx, :]                 

                # forward
                h_batch = r_u_batch.dot(W.T)          
                r_hat_batch = h_batch.dot(V_batch.T).diagonal()  
                e = R_norm[u_idx, i_idx] - r_hat_batch

                # gradients
                grad_V_batch = (- e[:, None] * h_batch + lam * V_batch) / len(batch)
                cp.clip(grad_V_batch, -1, 1, out=grad_V_batch)
                V[i_idx, :] -= alpha * grad_V_batch

                grad_W = (- (e[:, None] * V_batch).T.dot(r_u_batch) + lam * W) / len(batch)
                cp.clip(grad_W, -1, 1, out=grad_W)
                W -= alpha * grad_W

                batch_mse = float(cp.mean(e**2))
                history.append(batch_mse)
                pbar.set_postfix({'batch_mse': f"{batch_mse:.6f}"})

        # MSE total da época
        R_hat = (R_norm.dot(W.T)).dot(V.T)  
        diffs = (R_norm - R_hat)[M]
        epoch_mse = float(cp.mean(diffs**2))
        epoch_mse_history.append(epoch_mse)
        elapsed = time() - start_time
        print(f"Epoch {epoch:03d} | MSE={epoch_mse:.6f} | elapsed={elapsed:.1f}s")

        # Early stopping
        if epoch_mse < stop_mse or abs(last_mse - epoch_mse) < min_improvement:
            print(f"[train] Early stopping triggered at epoch {epoch}")
            break

        # Learning rate adaptativo
        if epoch_mse < last_mse - min_improvement:
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                alpha *= decay
                print(f"[train] Reducing learning rate to {alpha:.6f}")
                wait = 0

        last_mse = epoch_mse

    return W, V, history, epoch_mse_history, R_mean




# ---------- Recommendations ----------

def recommend_for_user(R, W, V, R_mean, user_idx, top_n=10):
    """
    Gera recomendações para um usuário, considerando normalização do treino.

    Args:
        R        : matriz original de ratings (U x N)
        W, V     : pesos treinados do autoencoder
        R_mean   : média de ratings por usuário (U x 1)
        user_idx : índice do usuário para recomendação
        top_n    : número de itens recomendados
    Returns:
        recommended_idx : índices dos top_n itens recomendados
        scores          : scores preditos correspondentes
    """
    # Normaliza o vetor do usuário
    r_u = (R[user_idx, :] - R_mean[user_idx]) / 4.5  # mesma normalização do treino

    # Calcula a codificação oculta
    h_u = r_u.dot(W.T)

    # Reconstrói os ratings (normalizados)
    r_hat_norm = h_u.dot(V.T)

    # Desnormaliza para a escala original
    r_hat = r_hat_norm * 5.0 + R_mean[user_idx]

    # Filtra apenas os itens não avaliados
    unrated_idx = cp.where(R[user_idx, :] == 0)[0]

    # Ordena e pega top_n
    recommended_idx = unrated_idx[cp.argsort(r_hat[unrated_idx])[::-1][:top_n]]

    return recommended_idx, r_hat[recommended_idx]


def per_user_mse(user_index, R, M, V, W, R_mean):
    """
    Calcula o MSE para um usuário, na escala original de ratings [0,5].

    Args:
        user_index (int) : índice do usuário
        R (cp.ndarray)  : matriz de ratings original
        M (cp.ndarray)  : máscara de observações
        V, W (cp.ndarray): pesos do autoencoder
        R_mean (cp.ndarray) : média dos ratings por usuário
    Returns:
        mse (float) : MSE do usuário
    """
    # Normaliza como no treino
    r_u = (R[user_index, :] - R_mean[user_index]) / 4.5
    h_u = r_u.dot(W.T)
    r_hat_norm = h_u.dot(V.T)
    r_hat = r_hat_norm * 5.0 + R_mean[user_index]  # desnormaliza

    diffs = (R[user_index, :] - r_hat)[M[user_index, :]]
    return float(cp.mean(diffs**2)) if diffs.size > 0 else float('nan')


# ---------- Função de recomendação em lote ----------
def recommend_for_users_batch(R, M, W, V, R_mean, user_indices, top_n=10):
    """
    Retorna recomendações e MSE para múltiplos usuários
    """
    results = []
    for uidx in user_indices:
        rec_idx, scores = recommend_for_user(R, W, V, R_mean, uidx, top_n)
        mse_u = per_user_mse(uidx, R, M, V, W, R_mean)
        results.append({
            'user_idx': uidx,
            'recommended_idx': rec_idx,
            'scores': scores,
            'mse': mse_u
        })
    return results

# ---------- Main ----------
def main():
    movies_csv = './Movies/movies.csv'
    ratings_csv = './Movies/ratings.csv'
    ratings, users_list, movies_list = load_ratings(ratings_csv)
    movie_titles = load_movies(movies_csv)
    R, M, user_to_idx, movie_to_idx = build_matrices(ratings, users_list, movies_list)
    idx_to_movieId = {idx: mid for mid, idx in movie_to_idx.items()}
    query_users = [40, 92, 123, 245, 312, 460, 514, 590]
    users_present = [user_to_idx[u] for u in query_users if u in user_to_idx]

    results_all_k = {}
    for k in KS:
        print("="*60)
        print(f"Treinando k={k}")
        W, V, history, epoch_mse_history, R_mean = train_autoencoder_batches(R, M, k)

        # Salvar histórico de MSE por batch
        with open(f"history_batch_k{k}.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['batch_idx','batch_mse'])
            for idx, val in enumerate(history,1):
                writer.writerow([idx,val])

        # Salvar histórico de MSE por época
        with open(f"history_epoch_k{k}.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch','mse'])
            for idx, val in enumerate(epoch_mse_history,1):
                writer.writerow([idx,val])

        # Recomendações e MSE por usuário
        results = recommend_for_users_batch(R, M, W, V, R_mean, users_present, top_n=5)

        # Salvar resultados por usuário em CSV
        with open(f"results_k{k}.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            header = ['userId', 'mse'] + [f'rec_{i+1}' for i in range(5)]
            writer.writerow(header)
            for res, orig_u in zip(results, query_users):
                rec_titles = [movie_titles.get(int(idx_to_movieId[int(idx)]), f"movieId_{int(idx)}") for idx in res['recommended_idx']]
                row = [orig_u, res['mse']] + rec_titles
                writer.writerow(row)

        # Salvar modelo treinado
        cp.savez(f"model_k{k}.npz", W=W, V=V)
        print(f"[save] modelo salvo em model_k{k}.npz")
        results_all_k[k] = results


if __name__ == "__main__":
    main()
