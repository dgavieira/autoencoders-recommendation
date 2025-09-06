#!/usr/bin/env python3
# autoencoder_sgd.py
# Implementação sem libs de alto nível (usa numpy e csv)
import csv
import numpy as np
import random
import math
from collections import defaultdict
from time import time

# ---------- PARÂMETROS ----------
ALPHA = 0.005      # taxa de aprendizado
LAMBDA = 0.0005    # regularização (10x menor que alpha)
KS = [50, 75, 100] # valores de k a testar
STOP_MSE = 0.1
MAX_EPOCHS = 200
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
# --------------------------------

def load_movies(movies_csv_path):
    """
    Lê movings.csv (assume colunas: movieId,title,...) e retorna dict movieId->title
    """
    movie_titles = {}
    with open(movies_csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = int(row['movieId'])
            movie_titles[mid] = row.get('title', str(mid))
    return movie_titles

def load_ratings(ratings_csv_path):
    """
    Lê ratings.csv (assume userId,movieId,rating,timestamp)
    Retorna:
      - ratings_list: lista de tuplas (userId, movieId, rating)
      - set_users, set_movies: conjuntos
    """
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
            except Exception as e:
                continue
            ratings.append((u, m, r))
            users.add(u)
            movies.add(m)
    return ratings, sorted(users), sorted(movies)

def build_matrices(ratings, users_list, movies_list):
    """
    Compacta userId/movieId para índices 0..U-1 / 0..N-1
    Constrói:
      - R: numpy array shape (U, N) com 0 para ausente
      - M: boolean mask (U, N) True onde rating existe
      - mappings: two dicts for id->index
    """
    user_to_idx = {u:i for i,u in enumerate(users_list)}
    movie_to_idx = {m:i for i,m in enumerate(movies_list)}
    U = len(users_list)
    N = len(movies_list)
    R = np.zeros((U, N), dtype=np.float32)
    M = np.zeros((U, N), dtype=bool)
    for (u, m, r) in ratings:
        if m in movie_to_idx and u in user_to_idx:
            ui = user_to_idx[u]
            mi = movie_to_idx[m]
            R[ui, mi] = r
            M[ui, mi] = True
    return R, M, user_to_idx, movie_to_idx

def initialize_weights(k, n):
    """
    W: (k, n) , V: (n, k)
    Inicialização uniforme [-1,1].
    """
    W = np.random.uniform(-1.0, 1.0, size=(k, n)).astype(np.float32)
    V = np.random.uniform(-1.0, 1.0, size=(n, k)).astype(np.float32)
    return W, V

def train_autoencoder_sgd(R, M, k, alpha=ALPHA, lam=LAMBDA, stop_mse=STOP_MSE, max_epochs=MAX_EPOCHS):
    """
    Treina o autoencoder linear com regra de atualização por amostra:
      Para cada (u,i) observado:
        h = W @ r_u        # r_u é o vetor de itens do usuário (inclui zeros)
        r_hat_i = v_i^\top h
        e = r_ui - r_hat_i
        grad_v_i = -e * h + lam * v_i
        grad_W = -e * (v_i[:,None] @ r_u[None,:]) + lam * W
        W <- W - alpha * grad_W
        v_i <- v_i - alpha * grad_v_i
    Retorna W, V, history (mse por época)
    """
    U, N = R.shape
    W, V = initialize_weights(k, N)
    # precompute list S of observed entries
    S = [(u,i) for u in range(U) for i in range(N) if M[u,i]]
    print(f"[train] U={U}, N={N}, #observed={len(S)}, k={k}")
    history = []
    epoch = 0
    start_time = time()
    while epoch < max_epochs:
        epoch += 1
        random.shuffle(S)
        # loop por amostras
        for (u,i) in S:
            r_u = R[u,:]  # vetor de itens do usuário (N,)
            # compute hidden representation h = W @ r_u  (k,)
            h = W.dot(r_u)  # shape (k,)
            v_i = V[i,:]    # shape (k,)
            r_hat_i = v_i.dot(h)
            e = R[u,i] - r_hat_i
            # gradients
            grad_v_i = -e * h + lam * v_i
            # grad_W: -e * outer(v_i, r_u) + lam * W
            # we'll compute update for all W (could optimize to only update cols where r_u != 0)
            outer = np.outer(v_i, r_u)  # shape (k, N)
            grad_W = -e * outer + lam * W
            # parameter updates
            V[i,:] = V[i,:] - alpha * grad_v_i
            W = W - alpha * grad_W
        # after epoch compute MSE over observed entries
        # compute H_all = W @ R.T? careful shapes: W (k,N), R[u,:] (N,) => h_u = W @ r_u
        # vectorized: H = W.dot(R.T)  # k x U
        H = W.dot(R.T)    # k x U
        R_hat = V.dot(H).T  # (U x N) ; V: n x k, H: k x U => V@H -> n x U, transpose -> U x n
        # compute mse only where M is True
        diffs = (R - R_hat)[M]
        mse = np.mean(diffs * diffs) if diffs.size > 0 else float('inf')
        history.append(mse)
        elapsed = time() - start_time
        print(f"Epoch {epoch:03d} | MSE={mse:.6f} | elapsed={elapsed:.1f}s")
        if mse < stop_mse:
            print(f"[train] Converged (MSE < {stop_mse}), stopping at epoch {epoch}")
            break
    return W, V, history

def recommend_for_user(user_index, R, M, V, W, top_k=5, movie_idx_to_id=None):
    """
    Gera previsões para um usuário (exclui filmes que já foram avaliados).
    Retorna lista de (movieId, score) ordenada decrescente.
    movie_idx_to_id: list or dict to map idx->movieId
    """
    r_u = R[user_index,:]
    h = W.dot(r_u)            # k,
    r_hat = V.dot(h)         # n,
    # mask out already rated
    rated_mask = M[user_index,:]
    r_hat_masked = np.where(rated_mask, -np.inf, r_hat)
    top_indices = np.argpartition(-r_hat_masked, range(top_k))[:top_k]
    top_indices_sorted = top_indices[np.argsort(-r_hat_masked[top_indices])]
    results = []
    for idx in top_indices_sorted:
        mid = movie_idx_to_id[idx] if movie_idx_to_id is not None else idx
        results.append((mid, float(r_hat[idx])))
    return results

def per_user_mse(user_index, R, M, V, W):
    """
    MSE das previsões do sistema para os filmes que o usuário avaliou.
    """
    r_u = R[user_index,:]
    h = W.dot(r_u)
    r_hat = V.dot(h)
    diffs = (r_u - r_hat)[M[user_index,:]]
    return float(np.mean(diffs * diffs)) if diffs.size > 0 else float('nan')

def main():
    movies_csv = './Movies/movies.csv'   # ajuste se necessário
    ratings_csv = './Movies/ratings.csv'
    ratings, users_list, movies_list = load_ratings(ratings_csv)
    movie_titles = load_movies(movies_csv)
    # garantir que consideramos apenas os movies_list (aqueles que aparecem em ratings)
    # construir matrizes
    R, M, user_to_idx, movie_to_idx = build_matrices(ratings, users_list, movies_list)
    # index -> original movieId
    idx_to_movieId = {idx: mid for mid, idx in movie_to_idx.items()}
    # Prepare list of specific users (os números referidos no enunciado: 40,92,...)
    query_users = [40, 92, 123, 245, 312, 460, 514, 590]
    # map them to indices if exist
    users_present = []
    for u in query_users:
        if u in user_to_idx:
            users_present.append((u, user_to_idx[u]))
        else:
            print(f"[warn] user {u} not found in data (skipping)")

    results_all_k = {}
    for k in KS:
        print("="*60)
        print(f"Treinando k={k}")
        W, V, history = train_autoencoder_sgd(R, M, k, alpha=ALPHA, lam=LAMBDA,
                                             stop_mse=STOP_MSE, max_epochs=MAX_EPOCHS)
        # para cada usuário pedido, gerar top-5 recomendações (apenas entre os filmes indexados)
        table_rows = []
        for (orig_u, uidx) in users_present:
            recs = recommend_for_user(uidx, R, M, V, W, top_k=5, movie_idx_to_id=idx_to_movieId)
            # map movieId to title when possible
            rec_titles = [movie_titles.get(mid, f"movieId_{mid}") for (mid,score) in recs]
            user_mse = per_user_mse(uidx, R, M, V, W)
            row = {
                'userId': orig_u,
                'recs': rec_titles,
                'mse': user_mse
            }
            table_rows.append(row)
        results_all_k[k] = {
            'W': W, 'V': V, 'history': history, 'table': table_rows
        }
        # salvar modelos num npz
        np.savez(f"model_k{k}.npz", W=W, V=V)
        print(f"[save] modelo salvo em model_k{k}.npz")

    # imprimir tabela final para o melhor k (menor MSE final)
    best_k = min(KS, key=lambda kk: results_all_k[kk]['history'][-1] if len(results_all_k[kk]['history'])>0 else float('inf'))
    print("\n" + "#"*40)
    print(f"Resumo final — melhor k = {best_k}")
    header = ["userId", "rec1", "rec2", "rec3", "rec4", "rec5", "user_mse"]
    print("\t".join(header))
    for row in results_all_k[best_k]['table']:
        recs = row['recs']
        recs = recs + [""]*(5-len(recs))
        print(f"{row['userId']}\t{recs[0]}\t{recs[1]}\t{recs[2]}\t{recs[3]}\t{recs[4]}\t{row['mse']:.6f}")

    # salvar a tabela em CSV
    out_csv = "recommendations_table_best_k.csv"
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in results_all_k[best_k]['table']:
            recs = row['recs']
            recs = recs + [""]*(5-len(recs))
            writer.writerow([row['userId']] + recs[:5] + [f"{row['mse']:.6f}"])
    print(f"[save] tabela salva em {out_csv}")

if __name__ == "__main__":
    main()
