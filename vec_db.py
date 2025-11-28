from typing import Dict, List, Annotated
import numpy as np
import os
import json
from heapq import heappush, heappushpop

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 70

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        if new_db:
            if db_size is None:
                raise ValueError("You need to provide the size of the database")
            # delete the old DB file if exists
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            self.generate_database(db_size)
    
    def generate_database(self, size: int) -> None:
        rng = np.random.default_rng(DB_SEED_NUMBER)
        vectors = rng.random((size, DIMENSION), dtype=np.float32)
        self._write_vectors_to_file(vectors)
        self._build_index()

    def _write_vectors_to_file(self, vectors: np.ndarray) -> None:
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='w+', shape=vectors.shape)
        mmap_vectors[:] = vectors[:]
        mmap_vectors.flush()

    def _get_num_records(self) -> int:
        return os.path.getsize(self.db_path) // (DIMENSION * ELEMENT_SIZE)

    def insert_records(self, rows: Annotated[np.ndarray, (int, 70)]):
        num_old_records = self._get_num_records()
        num_new_records = len(rows)
        full_shape = (num_old_records + num_new_records, DIMENSION)
        mmap_vectors = np.memmap(self.db_path, dtype=np.float32, mode='r+', shape=full_shape)
        mmap_vectors[num_old_records:] = rows
        mmap_vectors.flush()
        #TODO: might change to call insert in the index, if you need
        # if index exists, assign new IDs to centroids and append to the list files
        index_dir = self.index_path
        centroids_path = os.path.join(index_dir, "centroids.npy")
        if os.path.exists(centroids_path):
            centroids = np.load(centroids_path, mmap_mode='r')
            norms_cent = np.linalg.norm(centroids, axis=1)
            norms_cent[norms_cent == 0] = 1.0
            for i in range(num_old_records, num_old_records + num_new_records):
                vec = self.get_one_row(i)
                vnorm = np.linalg.norm(vec)
                if vnorm == 0:
                    vnorm = 1.0
                sims = (centroids @ vec) / (norms_cent * vnorm)
                cid = int(np.argmax(sims))
                list_file = os.path.join(index_dir, "lists", f"{cid:06d}.npy")
                if os.path.exists(list_file):
                    ids = np.load(list_file)
                    ids = np.concatenate([ids, np.array([i], dtype=np.uint32)])
                else:
                    ids = np.array([i], dtype=np.uint32)
                np.save(list_file, ids)
        else:
            # If no index exists, just rebuild full index
            self._build_index()

    def get_one_row(self, row_num: int) -> np.ndarray:
        # This function is only load one row in memory
        try:
            offset = row_num * DIMENSION * ELEMENT_SIZE
            mmap_vector = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(1, DIMENSION), offset=offset)
            return np.array(mmap_vector[0])
        except Exception as e:
            return f"An error occurred: {e}"

    def get_all_rows(self) -> np.ndarray:
        # Take care this load all the data in memory
        num_records = self._get_num_records()
        vectors = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))
        return np.array(vectors)
    
    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        nprobe = 66
        query = np.asarray(query, dtype=np.float32).reshape(DIMENSION)
        # load centroids
        centroids_path = os.path.join(self.index_path, "centroids.npy")
        centroids = np.load(centroids_path, mmap_mode='r') 
        qnorm = np.linalg.norm(query)
        if qnorm == 0:
            qnorm = 1.0
        cent_norms = np.linalg.norm(centroids, axis=1)
        cent_norms[cent_norms == 0] = 1.0
        sims_to_centroids = (centroids @ query) / (cent_norms * qnorm)  
        nearest_centroids = np.argsort(sims_to_centroids)[::-1][:nprobe]

        # Get candidates from nearest centroids
        heap = []
        lists_dir = os.path.join(self.index_path, "lists")
        for cid in nearest_centroids:
            list_file = os.path.join(lists_dir, f"{cid:06d}.npy")
            if not os.path.exists(list_file):
                continue
            ids = np.load(list_file, mmap_mode='r') 
            # Retrieve vectors and compute scores
            for vid in ids:
                vec = self.get_one_row(int(vid))
                # Compute cosine
                dot = float(np.dot(query, vec))
                denom = qnorm * (np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else 1.0)
                score = dot / denom
                # score = self._cal_score(query, vec)
                if len(heap) < top_k:
                    heappush(heap, (score, int(vid)))
                else:
                    # Replace smallest if current is larger
                    if score > heap[0][0]:
                        heappushpop(heap, (score, int(vid)))

        # return IDs sorted by score desc
        results = sorted(heap, key=lambda x: x[0], reverse=True)
        return [r[1] for r in results]


    def _cal_score(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity

    def _build_index(self, nlist: int = 1024, max_kmeans_samples: int = 200_000):
        # Sample vectors for kmeans
        num_records = self._get_num_records()
        sample_count = min(num_records, max_kmeans_samples)
        rng = np.random.default_rng(DB_SEED_NUMBER)
        sample_ids = rng.choice(num_records, size=sample_count, replace=False)
        sample_vectors = np.stack([self.get_one_row(int(i)) for i in sample_ids], axis=0).astype(np.float32)

        # Run kmeans on sample to produce centroids
        centroids = self._run_kmeans(sample_vectors, nlist=nlist, niter=12)

        # Assign all vectors to nearest centroid
        assignments = [[] for _ in range(nlist)]
        CHUNK = 10000
        for start in range(0, num_records, CHUNK):
            end = min(num_records, start + CHUNK)
            # Read chunk using memmap to avoid copying entire DB at once
            mmap_chunk = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))[start:end]
            # Normalize chunk and centroids for cosine similarity
            norms_chunk = np.linalg.norm(mmap_chunk, axis=1, keepdims=True)
            norms_chunk[norms_chunk == 0] = 1.0
            norms_cent = np.linalg.norm(centroids, axis=1, keepdims=True)
            norms_cent[norms_cent == 0] = 1.0
            sims = (mmap_chunk / norms_chunk) @ (centroids.T / norms_cent.T) 
            assigned = np.argmax(sims, axis=1)
            for i, a in enumerate(assigned):
                assignments[a].append(start + i)

        # Write index files
        index_dir = os.path.splitext(self.index_path)[0] + "_ivf_index"
        lists_dir = os.path.join(index_dir, "lists")
        os.makedirs(lists_dir, exist_ok=True)
        # Write centroids
        np.save(os.path.join(index_dir, "centroids.npy"), centroids)
        meta = {"nlist": nlist, "dimension": DIMENSION, "element_size": ELEMENT_SIZE}
        with open(os.path.join(index_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        # Write inverted lists
        for cid, id_list in enumerate(assignments):
            arr = np.array(id_list, dtype=np.uint32)
            np.save(os.path.join(lists_dir, f"{cid:06d}.npy"), arr)

        # update index path pointer
        self.index_path = index_dir


    def _run_kmeans(self, sample_vectors: np.ndarray, nlist: int, niter: int = 20):
        rng = np.random.default_rng(DB_SEED_NUMBER)
        # initialize centroids: pick random samples
        indices = rng.choice(len(sample_vectors), size=nlist, replace=False)
        centroids = sample_vectors[indices].astype(np.float32).copy()

        for it in range(niter):
            norms_cent = np.linalg.norm(centroids, axis=1, keepdims=True)
            # avoid dividing by zero
            norms_cent[norms_cent == 0] = 1.0
            # Calculate cosine similarity and pick nearest cluster
            cosine_similarity = sample_vectors @ (centroids.T / (norms_cent.T))  
            assign = np.argmax(cosine_similarity, axis=1)
            # Add vectors to their assigned centroid
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros((nlist,), dtype=np.int64)
            for i, a in enumerate(assign):
                counts[a] += 1
                new_centroids[a] += sample_vectors[i]
            # Update centroids with mean of assigned vectors
            for k in range(nlist):
                if counts[k] > 0:
                    new_centroids[k] /= counts[k]
                else:
                    # Reinitialize empty centroid
                    new_centroids[k] = sample_vectors[rng.integers(0, len(sample_vectors))]
            centroids = new_centroids
        return centroids.astype(np.float32)



