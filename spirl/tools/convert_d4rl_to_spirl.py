import os, h5py, numpy as np, gym, argparse
import d4rl  # pip install d4rl==1.1

def split_indices(n, train=0.95, val=0.025, test=0.025, seed=0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    n_train = int(n * train); n_val = int(n * val)
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def write_shards(root, split_name, trajs, shard_size=100):
    os.makedirs(os.path.join(root, split_name), exist_ok=True)
    for sid in range(0, len(trajs), shard_size):
        path = os.path.join(root, split_name, f"shard_{sid//shard_size:04d}.h5")
        with h5py.File(path, "w") as F:
            F.create_dataset("traj_per_file", data=min(shard_size, len(trajs) - sid))
            for i, (S, A) in enumerate(trajs[sid:sid+shard_size]):
                g = F.create_group(f"traj{i}")
                g.create_dataset("states",  data=S.astype(np.float32), compression="gzip")
                g.create_dataset("actions", data=A.astype(np.float32), compression="gzip")
                g.create_dataset("pad_mask", data=np.ones((S.shape[0],), dtype=np.float32))
                g.create_dataset("images", data=np.zeros((S.shape[0], 2, 2, 3), dtype=np.uint8))

def to_trajs(env_name):
    env = gym.make(env_name); ds = env.get_dataset()
    S, A = ds["observations"], ds["actions"]
    dones = ds.get("terminals"); timeouts = ds.get("timeouts")
    N = len(S); cut = np.zeros(N, dtype=bool)
    if dones is not None:    cut |= dones.astype(bool)
    if timeouts is not None: cut |= timeouts.astype(bool)

    trajs=[]; s=0
    for i in range(N-1):
        if cut[i]:
            if i+1-s>=2:
                trajs.append((S[s:i+1], A[s:i]))
            s=i+1
    if s < N-1 and N-s>=2:
        trajs.append((S[s:N], A[s:N-1]))
    return trajs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard_size", type=int, default=100)
    args = ap.parse_args()
    trajs = to_trajs(args.env)
    tr, va, te = split_indices(len(trajs), seed=args.seed)
    write_shards(args.out, "train", [trajs[i] for i in tr], args.shard_size)
    write_shards(args.out, "val",   [trajs[i] for i in va], args.shard_size)
    write_shards(args.out, "test",  [trajs[i] for i in te], args.shard_size)
    print(f"[OK] {args.env} → {args.out}  (traj={len(trajs)})")

if __name__ == "__main__":
    main()
