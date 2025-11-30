from typing import Dict, List, Annotated
import numpy as np
import os
import json
from heapq import heappush, heappushpop

DB_SEED_NUMBER = 42
ELEMENT_SIZE = np.dtype(np.float32).itemsize
DIMENSION = 64

class VecDB:
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index.dat", new_db = True, db_size = None) -> None:
        self.db_path = database_file_path
        self.index_path = index_file_path
        # self._build_index()  # For Building indexes one time
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

    def insert_records(self, rows: Annotated[np.ndarray, (int, 64)]):
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
    
    def get_rows_by_ids(self, ids: np.ndarray, max_block_size: int = 1_000) -> np.ndarray:
        # Retrieve multiple rows by their IDs.
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty((0, DIMENSION), dtype=np.float32)
        order = np.argsort(ids)
        ids_sorted = ids[order]

        # Build list of contiguous ranges
        ranges = []
        s = int(ids_sorted[0])
        prev = s
        for x in ids_sorted[1:]:
            x = int(x)
            if x == prev + 1:
                prev = x
                continue
            ranges.append((s, prev + 1))
            s = x
            prev = x
        ranges.append((s, prev + 1))

        # Open a single memmap for the DB once
        num_records = self._get_num_records()
        mmap_all = np.memmap(self.db_path, dtype=np.float32, mode='r', shape=(num_records, DIMENSION))

        pieces = []
        for (a, b) in ranges:
            length = b - a
            # If range is very large, chunk it to limit memory usage
            if length <= max_block_size:
                block = np.array(mmap_all[a:b], dtype=np.float32)
                pieces.append(block)
            else:
                p = a
                while p < b:
                    q = min(b, p + max_block_size)
                    pieces.append(np.array(mmap_all[p:q], dtype=np.float32))
                    p = q

        if len(pieces) == 0:
            out_sorted = np.empty((0, DIMENSION), dtype=np.float32)
        else:
            out_sorted = np.vstack(pieces) 

        # Map back to original order
        out = np.empty((ids.size, DIMENSION), dtype=np.float32)
        out[order] = out_sorted

        return out


    def _train_pq(self, sample_vectors: np.ndarray, M: int = 8, Ks: int = 256, niter: int = 20):
        D = sample_vectors.shape[1]
        if D % M != 0:
            raise ValueError("DIMENSION must be divisible by M")
        subdim = D // M
        rng = np.random.default_rng(DB_SEED_NUMBER)

        codebooks = np.zeros((M, Ks, subdim), dtype=np.float32)

        for m in range(M):
            sub = sample_vectors[:, m*subdim:(m+1)*subdim]
            # Initialize centroids by sampling
            idx = rng.choice(len(sub), size=Ks, replace=False)
            centroids = sub[idx].astype(np.float32).copy()

            for it in range(niter):
                dots = sub @ centroids.T                       
                sub_sq = np.sum(sub*sub, axis=1, keepdims=True)   
                cent_sq = np.sum(centroids*centroids, axis=1)     
                dists = sub_sq - 2.0 * dots + cent_sq[None, :]   
                assign = np.argmin(dists, axis=1)
                new_centroids = np.zeros_like(centroids)
                counts = np.zeros((Ks,), dtype=np.int64)
                for i, a in enumerate(assign):
                    counts[a] += 1
                    new_centroids[a] += sub[i]
                for k in range(Ks):
                    if counts[k] > 0:
                        new_centroids[k] /= counts[k]
                    else:
                        new_centroids[k] = sub[rng.integers(0, len(sub))]
                centroids = new_centroids
            codebooks[m] = centroids
        return codebooks.astype(np.float32)


    def _encode_pq_codes_for_ids(self, ids: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
        # Encode vectors with given IDs into PQ codes.
        ids = np.asarray(ids, dtype=np.int64)
        if ids.size == 0:
            return np.empty((0, codebooks.shape[0]), dtype=np.uint8)

        vecs = self.get_rows_by_ids(ids)  
        M, Ks, subdim = codebooks.shape
        codes = np.empty((len(ids), M), dtype=np.uint8)

        for m in range(M):
            sub = vecs[:, m*subdim:(m+1)*subdim]   
            cent = codebooks[m]                    
            dots = sub @ cent.T                    
            sub_sq = np.sum(sub*sub, axis=1, keepdims=True)  
            cent_sq = np.sum(cent*cent, axis=1)              
            dists = sub_sq - 2.0 * dots + cent_sq[None, :]   
            assign = np.argmin(dists, axis=1)
            codes[:, m] = assign.astype(np.uint8)
        return codes


    def retrieve(self, query: Annotated[np.ndarray, (1, DIMENSION)], top_k=5):
        if (self._get_num_records() / 1_000_000) <= 10:
            nprobe = 2
            rerank_k = 70
        else:
            nprobe = 2
            rerank_k = 700
        index_dir = self.index_path
        centroids = np.load(os.path.join(index_dir, "centroids.npy"))
        pq_dir = os.path.join(index_dir, "pq")
        codebooks_path = os.path.join(pq_dir, "pq_codebooks.npy")
        if not os.path.exists(codebooks_path):
            raise RuntimeError("PQ index not found")

        # M is number of subquantizers, Ks is number of centroids per subquantizer, subdim is dimension / M
        pq_codebooks = np.load(codebooks_path)
        M, Ks, subdim = pq_codebooks.shape

        q = np.asarray(query, dtype=np.float32).reshape(DIMENSION)
        qnorm = np.linalg.norm(q)
        if qnorm == 0:
            qnorm = 1.0

        # Choose centroids by cosine similarity
        cent_norms = np.linalg.norm(centroids, axis=1)
        cent_norms[cent_norms == 0] = 1.0
        sims_to_centroids = (centroids @ q) / (cent_norms * qnorm)
        nearest_centroids = np.argsort(sims_to_centroids)[::-1][:nprobe]

        lists_dir = os.path.join(index_dir, "lists")
        codes_dir = os.path.join(pq_dir, "codes")
        heap = []

        # Measure distances using ADC (euclidean): ||q||² - 2 q·c + ||c||²
        lut_local = np.empty((M, Ks), dtype=np.float32)
        for m in range(M):
            q_sub = q[m*subdim:(m+1)*subdim]
            cent = pq_codebooks[m] 
            q_sq = np.sum(q_sub * q_sub)
            dots = cent @ q_sub     
            cent_sq = np.sum(cent * cent, axis=1)
            lut_local[m] = q_sq - 2.0 * dots + cent_sq

        # Process each centroid's list
        for cid in nearest_centroids:
            list_file = os.path.join(lists_dir, f"{cid:06d}.npy")
            codes_file = os.path.join(codes_dir, f"{cid:06d}_codes.npy")

            ids_mm = np.load(list_file)
            codes_mm = np.load(codes_file)

            # Compute approximate distances for the list
            approx_dists = np.sum(lut_local[np.arange(M)[:,None], codes_mm.T], axis=0)

            # Maintain top-n (rerank_k) in heap
            for vid, dist in zip(ids_mm, approx_dists):
                neg = -dist
                if len(heap) < rerank_k:
                    heappush(heap, (neg, int(vid)))
                else:
                    if neg > heap[0][0]:
                        heappushpop(heap, (neg, int(vid)))

        if not heap:
            return []

        # Sort ids by distance ascending
        heap_items = sorted([(-d, vid) for d, vid in heap], key=lambda x: x[0])
        rerank_ids = np.array([vid for _, vid in heap_items], dtype=np.int64)
        rerank_vecs = []

        # Get full vectors for rerank candidates
        for x in rerank_ids:
            rerank_vec = self.get_one_row(x)
            # Compute cosine similarity
            score = self._cal_score(q, rerank_vec)
            rerank_vecs.append((score, x))
        rerank_vecs.sort(key=lambda x: x[0], reverse=True)
        top_ids = [int(t[1]) for t in rerank_vecs[:top_k]]
        return top_ids


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
        index_dir = os.path.splitext(self.index_path)[0]
        lists_dir = os.path.join(index_dir, "lists")
        os.makedirs(lists_dir, exist_ok=True)
        # Write centroids
        np.save(os.path.join(index_dir, "centroids.npy"), centroids)
        meta = {"nlist": nlist, "dimension": DIMENSION, "element_size": ELEMENT_SIZE}
        with open(os.path.join(index_dir, "meta.json"), "w") as f:
            json.dump(meta, f)
        # Write inverted lists (IDs)
        for cid, id_list in enumerate(assignments):
            arr = np.array(id_list, dtype=np.uint32)
            np.save(os.path.join(lists_dir, f"{cid:06d}.npy"), arr)

        # Train PQ on sample vectors
        M = 8
        Ks = 256
        pq_codebooks = self._train_pq(sample_vectors, M=M, Ks=Ks, niter=12)

        pq_dir = os.path.join(index_dir, "pq")
        os.makedirs(pq_dir, exist_ok=True)
        np.save(os.path.join(pq_dir, "pq_codebooks.npy"), pq_codebooks)
        with open(os.path.join(pq_dir, "pq_meta.json"), "w") as f:
            json.dump({"M": M, "Ks": Ks, "subdim": DIMENSION // M}, f)

        codes_dir = os.path.join(pq_dir, "codes")
        os.makedirs(codes_dir, exist_ok=True)
        for cid, id_list in enumerate(assignments):
            ids_arr = np.array(id_list, dtype=np.int64)
            if ids_arr.size == 0:
                np.save(os.path.join(codes_dir, f"{cid:06d}_codes.npy"), np.empty((0, M), dtype=np.uint8))
                continue
            codes = self._encode_pq_codes_for_ids(ids_arr, pq_codebooks) 
            np.save(os.path.join(codes_dir, f"{cid:06d}_codes.npy"), codes)

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
