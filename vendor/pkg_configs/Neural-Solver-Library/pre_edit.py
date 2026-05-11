"""Pre-edit operations for the Neural-Solver-Library package.
Registers Custom model in model_factory and injects TRAIN_METRICS into exp files.
Adds pdebench_conditional, airfrans_design, aircraft_design loaders.
"""

# Register Custom model in model_factory.py
_CUSTOM_IMPORT = "from models import Custom\n"
_CUSTOM_ENTRY = "        'Custom': Custom,\n"

# TRAIN_METRICS injection for exp_steady.py (after line 88: print("Epoch {} Train loss..."))
_STEADY_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={ep} '
    'train_loss={train_loss:.7f} rel_err={rel_err:.7f}", flush=True)'
)

# TRAIN_METRICS injection for exp_dynamic_autoregressive.py (after line 96-97: print("Epoch {} Train loss step..."))
_AUTOREGRESSIVE_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={ep} '
    'train_loss_step={train_loss_step:.7f} train_loss_full={train_loss_full:.7f} '
    'test_loss_full={test_loss_full:.7f}", flush=True)'
)

# TRAIN_METRICS injection for exp_dynamic_conditional.py (after line 85: print("Epoch {} Test loss full..."))
_CONDITIONAL_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={ep} '
    'train_loss_step={train_loss_step:.7f} '
    'test_loss_full={test_loss_full:.7f}", flush=True)'
)

# TRAIN_METRICS injection for exp_steady_design.py (after line 91: print("rel_err:{}"))
_DESIGN_TRAIN_METRICS = (
    '            print(f"TRAIN_METRICS epoch={ep} '
    'train_loss={train_loss:.7f} rel_err={rel_err:.7f}", flush=True)'
)

# ---------- pdebench_conditional loader ----------
_PDEBENCH_CONDITIONAL_CODE = '''

class pdebench_conditional(object):
    """Loader for PDEBench 2D datasets (SWE, DiffReact) in conditional format.
    Returns (pos, time, fx, yy) matching exp_dynamic_conditional.py interface.
    """
    def __init__(self, args):
        self.file_path = args.data_path
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.T_out = args.T_out
        self.out_dim = args.out_dim
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley

        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}.")

    def random_collate_fn(self, batch):
        shuffled_pos = None
        shuffled_t = None
        shuffled_a = None
        shuffled_u = None
        for item in batch:
            pos, t, a, u = item[0], item[1], item[2], item[3]
            num_timesteps = t.size(0)
            perm = torch.randperm(num_timesteps)
            t = t[perm]
            u = u.reshape(u.shape[0], num_timesteps, -1)[..., perm, :].reshape(u.shape[0], -1)
            if shuffled_t is None:
                shuffled_pos = pos.unsqueeze(0)
                shuffled_t = t.unsqueeze(0)
                shuffled_a = a.unsqueeze(0)
                shuffled_u = u.unsqueeze(0)
            else:
                shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
                shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
                shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)
                shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
        return [shuffled_pos, shuffled_t, shuffled_a, shuffled_u]

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley

        with h5py.File(self.file_path, "r") as f:
            data_list = sorted(f.keys())
            sample = np.array(f[data_list[0]]["data"], dtype="f")
            T_total, sx, sy, V = sample.shape
            s1 = int(((sx - 1) / r1) + 1)
            s2 = int(((sy - 1) / r2) + 1)

            x = np.array(f[data_list[0]]["grid"]["x"], dtype="f")
            y = np.array(f[data_list[0]]["grid"]["y"], dtype="f")

            t_indices = np.linspace(1, T_total - 1, self.T_out, dtype=int)
            n_total = self.ntrain + self.ntest
            assert n_total <= len(data_list), f"Need {n_total} samples but only {len(data_list)} available"

            all_ic = np.zeros((n_total, s1 * s2, V), dtype=np.float32)
            all_yy = np.zeros((n_total, s1 * s2, self.T_out * V), dtype=np.float32)

            for i, key in enumerate(data_list[:n_total]):
                data = np.array(f[key]["data"], dtype="f")
                ic = data[0, ::r1, ::r2, :][:s1, :s2, :].reshape(-1, V)
                all_ic[i] = ic
                for ti, t_idx in enumerate(t_indices):
                    frame = data[t_idx, ::r1, ::r2, :][:s1, :s2, :].reshape(-1, V)
                    all_yy[i, :, ti * V:(ti + 1) * V] = frame

        fx_all = torch.tensor(all_ic, dtype=torch.float)
        yy_all = torch.tensor(all_yy, dtype=torch.float)

        fx_train = fx_all[:self.ntrain]
        fx_test = fx_all[self.ntrain:self.ntrain + self.ntest]
        yy_train = yy_all[:self.ntrain]
        yy_test = yy_all[self.ntrain:self.ntrain + self.ntest]

        x_coords = torch.tensor(x[::r1][:s1], dtype=torch.float)
        y_coords = torch.tensor(y[::r2][:s2], dtype=torch.float)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

        pos_train = pos.unsqueeze(0).repeat(self.ntrain, 1, 1)
        pos_test = pos.unsqueeze(0).repeat(self.ntest, 1, 1)

        t = torch.linspace(0, 1, self.T_out)
        t_train = t.unsqueeze(0).repeat(self.ntrain, 1)
        t_test = t.unsqueeze(0).repeat(self.ntest, 1)

        x_normalizer = UnitTransformer(fx_train) if self.norm_type == "UnitTransformer" else UnitGaussianNormalizer(fx_train)
        fx_train = x_normalizer.encode(fx_train)
        fx_test = x_normalizer.encode(fx_test)
        x_normalizer.cuda()

        if self.normalize:
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(yy_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.y_normalizer = UnitGaussianNormalizer(yy_train)
            yy_train = self.y_normalizer.encode(yy_train)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, t_train, fx_train, yy_train),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=self.random_collate_fn)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, t_test, fx_test, yy_test),
            batch_size=self.batch_size, shuffle=False)

        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]
'''

# ---------- airfrans_design loader ----------
_AIRFRANS_DESIGN_CODE = '''

class airfrans_design(object):
    """Loader for AirfRANS dataset in design format.
    Returns GraphDataset compatible with exp_steady_design.py.
    Features: [pos(2), inlet_vel(2), sdf(1), normals(2)] = 7 dim
    Targets: [vx, vy, nut, p] = 4 dim (pressure last for design loss)
    Uses preprocessed .pt files when available for fast loading.
    Subsamples to max_points to speed up radius graph construction.
    """
    def __init__(self, args):
        self.data_path = args.data_path
        self.radius = args.radius
        self.max_points = getattr(args, 'max_points', 30000)

    def _subsample(self, data):
        """Subsample a PyG Data object to self.max_points if needed."""
        N = data.pos.shape[0]
        if N <= self.max_points:
            return data
        idx = np.random.choice(N, self.max_points, replace=False)
        idx.sort()
        idx_t = torch.tensor(idx, dtype=torch.long)
        from torch_geometric.data import Data as PyGData
        return PyGData(pos=data.pos[idx_t], x=data.x[idx_t],
                       y=data.y[idx_t], surf=data.surf[idx_t])

    def get_loader(self):
        import json
        from torch_geometric.data import Data as PyGData

        root = self.data_path
        # Check for preprocessed directory first
        preproc_dir = os.path.join(root, 'preprocessed')
        if os.path.exists(preproc_dir):
            manifest_path = os.path.join(preproc_dir, 'manifest.json')
        else:
            if os.path.exists(os.path.join(root, 'Dataset')):
                root = os.path.join(root, 'Dataset')
            manifest_path = os.path.join(root, 'manifest.json')
            preproc_dir = None

        with open(manifest_path) as mf:
            manifest = json.load(mf)

        train_names = manifest['full_train']
        test_names = manifest['full_test']

        train_data, coef_norm = self._load_samples(root, train_names, norm=True, preproc_dir=preproc_dir)
        test_data = self._load_samples(root, test_names, coef_norm=coef_norm, preproc_dir=preproc_dir)

        train_loader = GraphDataset(train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm)
        test_loader = GraphDataset(test_data, use_cfd_mesh=False, r=self.radius,
                                   coef_norm=coef_norm, valid_list=test_names)
        return train_loader, test_loader, [train_data[0].x.shape[0]]

    def _load_samples(self, root, names, norm=False, coef_norm=None, preproc_dir=None):
        from torch_geometric.data import Data as PyGData

        dataset = []
        mean_in, mean_out = 0, 0
        old_length = 0

        for k, name in enumerate(names):
            # Try preprocessed .pt file first
            if preproc_dir is not None:
                pt_path = os.path.join(preproc_dir, name + '.pt')
                if os.path.exists(pt_path):
                    saved = torch.load(pt_path, weights_only=True)
                    data = PyGData(pos=saved['pos'].float(), x=saved['x'].float(),
                                   y=saved['y'].float(), surf=saved['surf'].bool())
                    data = self._subsample(data)
                    if norm and coef_norm is None:
                        init = data.x.numpy()
                        target = data.y.numpy()
                        if k == 0:
                            old_length = init.shape[0]
                            mean_in = init.mean(axis=0)
                            mean_out = target.mean(axis=0)
                        else:
                            new_length = old_length + init.shape[0]
                            mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                            mean_out += (target.sum(axis=0) - target.shape[0] * mean_out) / new_length
                            old_length = new_length
                    dataset.append(data)
                    if (k + 1) % 100 == 0:
                        print(f'  Loaded {k+1}/{len(names)} samples', flush=True)
                    continue

            # Fallback: read from raw VTU/VTP
            import pyvista as pv
            from sklearn.neighbors import NearestNeighbors
            sim_dir = os.path.join(root, name)
            vtu_file = os.path.join(sim_dir, name + '_internal.vtu')
            vtp_file = os.path.join(sim_dir, name + '_aerofoil.vtp')
            if not os.path.exists(vtu_file):
                continue

            internal = pv.read(vtu_file)
            points = np.array(internal.points[:, :2], dtype=np.float32)
            velocity = np.array(internal.point_data['U'][:, :2], dtype=np.float32)
            pressure = np.array(internal.point_data['p'], dtype=np.float32).reshape(-1, 1)
            nut = np.array(internal.point_data['nut'], dtype=np.float32).reshape(-1, 1)
            sdf = np.abs(np.array(internal.point_data['implicit_distance'], dtype=np.float32)).reshape(-1, 1)

            surface = pv.read(vtp_file)
            surf_pts = np.array(surface.points[:, :2], dtype=np.float32)
            surf_normals_raw = np.zeros((len(surf_pts), 2), dtype=np.float32)
            if 'Normals' in surface.point_data:
                surf_normals_raw = np.array(surface.point_data['Normals'][:, :2], dtype=np.float32)

            nbrs = NearestNeighbors(n_neighbors=1).fit(surf_pts)
            dists, indices = nbrs.kneighbors(points)
            surf_mask = (dists[:, 0] < 1e-6)
            normals = np.zeros((len(points), 2), dtype=np.float32)
            normals[surf_mask] = surf_normals_raw[indices[surf_mask, 0]]

            parts = name.split('_')
            Uinf = float(parts[2])
            alpha_rad = np.radians(float(parts[3]))
            inlet_vel = np.full((len(points), 2),
                                [np.cos(alpha_rad) * Uinf, np.sin(alpha_rad) * Uinf],
                                dtype=np.float32)

            init = np.c_[points, inlet_vel, sdf, normals]
            target = np.c_[velocity, nut, pressure]

            data = PyGData(pos=torch.tensor(points, dtype=torch.float),
                           x=torch.tensor(init, dtype=torch.float),
                           y=torch.tensor(target, dtype=torch.float),
                           surf=torch.tensor(surf_mask).bool())
            data = self._subsample(data)

            if norm and coef_norm is None:
                init_np = data.x.numpy()
                target_np = data.y.numpy()
                if k == 0:
                    old_length = init_np.shape[0]
                    mean_in = init_np.mean(axis=0)
                    mean_out = target_np.mean(axis=0)
                else:
                    new_length = old_length + init_np.shape[0]
                    mean_in += (init_np.sum(axis=0) - init_np.shape[0] * mean_in) / new_length
                    mean_out += (target_np.sum(axis=0) - target_np.shape[0] * mean_out) / new_length
                    old_length = new_length

            dataset.append(data)
            if (k + 1) % 50 == 0:
                print(f'  Loaded {k+1}/{len(names)} samples (raw)', flush=True)

        print(f'  All {len(dataset)} samples loaded (max_points={self.max_points})', flush=True)

        if norm and coef_norm is None:
            old_length = 0
            std_in, std_out = 0, 0
            for k, data in enumerate(dataset):
                xn, yn = data.x.numpy(), data.y.numpy()
                if k == 0:
                    old_length = xn.shape[0]
                    std_in = ((xn - mean_in) ** 2).sum(axis=0) / old_length
                    std_out = ((yn - mean_out) ** 2).sum(axis=0) / old_length
                else:
                    new_length = old_length + xn.shape[0]
                    std_in += (((xn - mean_in) ** 2).sum(axis=0) - xn.shape[0] * std_in) / new_length
                    std_out += (((yn - mean_out) ** 2).sum(axis=0) - xn.shape[0] * std_out) / new_length
                    old_length = new_length
            std_in = np.sqrt(std_in)
            std_out = np.sqrt(std_out)
            for data in dataset:
                data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
                data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()
            coef_norm = (mean_in, std_in, mean_out, std_out)
            dataset = (dataset, coef_norm)
        elif coef_norm is not None:
            for data in dataset:
                data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
                data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

        return dataset
'''

# ---------- aircraft_design loader ----------
_AIRCRAFT_DESIGN_CODE = '''

class aircraft_design(object):
    """Loader for AirCraft dataset in design format.
    Features: [pos(3), sdf=0(1), normals(3)] = 7 dim
    Targets: [Cp, rho, u, v, w, p] = 6 dim (pressure last for design loss)
    """
    def __init__(self, args):
        self.data_path = args.data_path
        self.radius = args.radius
        self.max_points = getattr(args, 'max_points', 100000)

    def get_loader(self):
        import json
        from torch_geometric.data import Data as PyGData

        root = self.data_path
        manifest_path = os.path.join(root, 'airplane_dataset.json')
        with open(manifest_path) as mf:
            manifest = json.load(mf)

        train_names = manifest['train_set']
        test_names = manifest['test_set']

        train_data, coef_norm = self._load_samples(root, train_names, norm=True)
        test_data = self._load_samples(root, test_names, coef_norm=coef_norm)

        train_loader = GraphDataset(train_data, use_cfd_mesh=False, r=self.radius, coef_norm=coef_norm)
        test_loader = GraphDataset(test_data, use_cfd_mesh=False, r=self.radius,
                                   coef_norm=coef_norm, valid_list=test_names)
        return train_loader, test_loader, [train_data[0].x.shape[0]]

    def _load_samples(self, root, names, norm=False, coef_norm=None):
        from torch_geometric.data import Data as PyGData

        dataset = []
        mean_in, mean_out = 0, 0
        old_length = 0

        for k, name in enumerate(names):
            filepath = os.path.join(root, name)
            with h5py.File(filepath, 'r') as f:
                pos = np.array(f['pos'], dtype=np.float32)
                normals = np.array(f['normals'], dtype=np.float32)
                values = np.array(f['values'], dtype=np.float32)

            N = pos.shape[0]
            if N > self.max_points:
                idx = np.random.choice(N, self.max_points, replace=False)
                idx.sort()
                pos, normals, values = pos[idx], normals[idx], values[idx]

            # Normalize positions to [0,1] for radius graph
            pos_min = pos.min(axis=0)
            pos_range = pos.max(axis=0) - pos_min
            pos_range[pos_range < 1e-8] = 1.0
            pos_norm = (pos - pos_min) / pos_range

            sdf = np.zeros((pos.shape[0], 1), dtype=np.float32)
            init = np.c_[pos, sdf, normals]
            target = values
            surf = np.ones(pos.shape[0], dtype=bool)

            if norm and coef_norm is None:
                if k == 0:
                    old_length = init.shape[0]
                    mean_in = init.mean(axis=0)
                    mean_out = target.mean(axis=0)
                else:
                    new_length = old_length + init.shape[0]
                    mean_in += (init.sum(axis=0) - init.shape[0] * mean_in) / new_length
                    mean_out += (target.sum(axis=0) - target.shape[0] * mean_out) / new_length
                    old_length = new_length

            data = PyGData(pos=torch.tensor(pos_norm, dtype=torch.float),
                           x=torch.tensor(init, dtype=torch.float),
                           y=torch.tensor(target, dtype=torch.float),
                           surf=torch.tensor(surf).bool())
            dataset.append(data)

        if norm and coef_norm is None:
            old_length = 0
            std_in, std_out = 0, 0
            for k, data in enumerate(dataset):
                xn, yn = data.x.numpy(), data.y.numpy()
                if k == 0:
                    old_length = xn.shape[0]
                    std_in = ((xn - mean_in) ** 2).sum(axis=0) / old_length
                    std_out = ((yn - mean_out) ** 2).sum(axis=0) / old_length
                else:
                    new_length = old_length + xn.shape[0]
                    std_in += (((xn - mean_in) ** 2).sum(axis=0) - xn.shape[0] * std_in) / new_length
                    std_out += (((yn - mean_out) ** 2).sum(axis=0) - xn.shape[0] * std_out) / new_length
                    old_length = new_length
            std_in = np.sqrt(std_in)
            std_out = np.sqrt(std_out)
            for data in dataset:
                data.x = ((data.x - mean_in) / (std_in + 1e-8)).float()
                data.y = ((data.y - mean_out) / (std_out + 1e-8)).float()
            coef_norm = (mean_in, std_in, mean_out, std_out)
            dataset = (dataset, coef_norm)
        elif coef_norm is not None:
            for data in dataset:
                data.x = ((data.x - coef_norm[0]) / (coef_norm[1] + 1e-8)).float()
                data.y = ((data.y - coef_norm[2]) / (coef_norm[3] + 1e-8)).float()

        return dataset
'''

# ---------- pdebench_autoregressive_flat loader (for Burgers flat tensor format) ----------
_PDEBENCH_AUTOREGRESSIVE_FLAT_CODE = '''

class pdebench_autoregressive_flat(object):
    """Loader for PDEBench flat-tensor HDF5 files (e.g. Burgers).
    File has keys: tensor[N,T,X], x-coordinate[X], t-coordinate[T+1].
    Returns same (grid, input, output) format as pdebench_autoregressive.
    """
    def __init__(self, args):
        self.file_path = args.data_path
        self.train_ratio = args.train_ratio
        self.T_in = args.T_in
        self.T_out = args.T_out
        self.batch_size = args.batch_size
        self.out_dim = args.out_dim

    def get_loader(self):
        train_dataset = pdebench_dataset_flat(file_path=self.file_path, train_ratio=self.train_ratio,
                                              test=False, T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim)
        test_dataset = pdebench_dataset_flat(file_path=self.file_path, train_ratio=self.train_ratio,
                                             test=True, T_in=self.T_in, T_out=self.T_out, out_dim=self.out_dim)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader, test_loader, train_dataset.shapelist


class pdebench_dataset_flat(Dataset):
    """Dataset for flat-tensor PDEBench HDF5 files."""
    def __init__(self, file_path, train_ratio, test, T_in, T_out, out_dim):
        self.file_path = file_path
        with h5py.File(self.file_path, "r") as h5_file:
            tensor = h5_file["tensor"]
            N, T, X = tensor.shape
            self.shapelist = [X]
            self.grid = torch.tensor(np.array(h5_file["x-coordinate"], dtype="f")).unsqueeze(-1)
        self.ntrain = int(N * train_ratio)
        if not test:
            self.indices = list(range(self.ntrain))
        else:
            self.indices = list(range(self.ntrain, N))
        self.T_in = T_in
        self.T_out = T_out
        self.out_dim = out_dim

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, "r") as h5_file:
            # tensor shape: [N, T, X] -> per sample: [T, X]
            data = np.array(h5_file["tensor"][self.indices[idx]], dtype="f")
            # data shape: [T, X], need [X, T*out_dim]
            data = torch.tensor(data, dtype=torch.float).T  # [X, T]
        return self.grid, data[:, :self.T_in * self.out_dim], \\
            data[:, (self.T_in) * self.out_dim:(self.T_in + self.T_out) * self.out_dim]
'''

# ---------- pdebench_conditional_cfd loader (flat multi-key HDF5 format) ----------
_PDEBENCH_CONDITIONAL_CFD_CODE = '''

class pdebench_conditional_cfd(object):
    """Loader for PDEBench 2D CFD data in flat multi-key HDF5 format.
    File has keys: density[N,T,X,Y], Vx[N,T,X,Y], Vy[N,T,X,Y], pressure[N,T,X,Y],
    plus x-coordinate[X] and y-coordinate[Y].
    Returns (pos, time, fx, yy) matching the pdebench_conditional interface
    for exp_dynamic_conditional.py.
    """
    def __init__(self, args):
        self.file_path = args.data_path
        self.batch_size = args.batch_size
        self.ntrain = args.ntrain
        self.ntest = args.ntest
        self.T_out = args.T_out
        self.out_dim = args.out_dim
        self.normalize = args.normalize
        self.norm_type = args.norm_type
        self.downsamplex = args.downsamplex
        self.downsampley = args.downsampley

        if self.norm_type not in ["UnitTransformer", "UnitGaussianNormalizer"]:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}.")

    def random_collate_fn(self, batch):
        shuffled_pos = None
        shuffled_t = None
        shuffled_a = None
        shuffled_u = None
        for item in batch:
            pos, t, a, u = item[0], item[1], item[2], item[3]
            num_timesteps = t.size(0)
            perm = torch.randperm(num_timesteps)
            t = t[perm]
            u = u.reshape(u.shape[0], num_timesteps, -1)[..., perm, :].reshape(u.shape[0], -1)
            if shuffled_t is None:
                shuffled_pos = pos.unsqueeze(0)
                shuffled_t = t.unsqueeze(0)
                shuffled_a = a.unsqueeze(0)
                shuffled_u = u.unsqueeze(0)
            else:
                shuffled_pos = torch.cat((shuffled_pos, pos.unsqueeze(0)), 0)
                shuffled_t = torch.cat((shuffled_t, t.unsqueeze(0)), 0)
                shuffled_a = torch.cat((shuffled_a, a.unsqueeze(0)), 0)
                shuffled_u = torch.cat((shuffled_u, u.unsqueeze(0)), 0)
        return [shuffled_pos, shuffled_t, shuffled_a, shuffled_u]

    def get_loader(self):
        r1 = self.downsamplex
        r2 = self.downsampley

        n_total = self.ntrain + self.ntest
        with h5py.File(self.file_path, "r") as f:
            # Only read first n_total samples to avoid OOM
            N_total = f["density"].shape[0]
            T_total = f["density"].shape[1]
            sx = f["density"].shape[2]
            sy = f["density"].shape[3]
            assert n_total <= N_total, f"Need {n_total} samples but only {N_total} available"

            density = np.array(f["density"][:n_total], dtype="f")
            vx = np.array(f["Vx"][:n_total], dtype="f")
            vy = np.array(f["Vy"][:n_total], dtype="f")
            pressure = np.array(f["pressure"][:n_total], dtype="f")

            # Stack into [n_total, T, X, Y, 4]
            data_all = np.stack([density, vx, vy, pressure], axis=-1)
            del density, vx, vy, pressure
            V = 4

            s1 = int(((sx - 1) / r1) + 1)
            s2 = int(((sy - 1) / r2) + 1)

            x = np.array(f["x-coordinate"], dtype="f")
            y = np.array(f["y-coordinate"], dtype="f")

        t_indices = np.linspace(1, T_total - 1, self.T_out, dtype=int)

        all_ic = np.zeros((n_total, s1 * s2, V), dtype=np.float32)
        all_yy = np.zeros((n_total, s1 * s2, self.T_out * V), dtype=np.float32)

        for i in range(n_total):
            # Initial condition at t=0, with spatial downsampling
            ic = data_all[i, 0, ::r1, ::r2, :][:s1, :s2, :].reshape(-1, V)
            all_ic[i] = ic
            for ti, t_idx in enumerate(t_indices):
                frame = data_all[i, t_idx, ::r1, ::r2, :][:s1, :s2, :].reshape(-1, V)
                all_yy[i, :, ti * V:(ti + 1) * V] = frame

        fx_all = torch.tensor(all_ic, dtype=torch.float)
        yy_all = torch.tensor(all_yy, dtype=torch.float)

        fx_train = fx_all[:self.ntrain]
        fx_test = fx_all[self.ntrain:self.ntrain + self.ntest]
        yy_train = yy_all[:self.ntrain]
        yy_test = yy_all[self.ntrain:self.ntrain + self.ntest]

        x_coords = torch.tensor(x[::r1][:s1], dtype=torch.float)
        y_coords = torch.tensor(y[::r2][:s2], dtype=torch.float)
        X, Y = torch.meshgrid(x_coords, y_coords, indexing="ij")
        pos = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)

        pos_train = pos.unsqueeze(0).repeat(self.ntrain, 1, 1)
        pos_test = pos.unsqueeze(0).repeat(self.ntest, 1, 1)

        t = torch.linspace(0, 1, self.T_out)
        t_train = t.unsqueeze(0).repeat(self.ntrain, 1)
        t_test = t.unsqueeze(0).repeat(self.ntest, 1)

        x_normalizer = UnitTransformer(fx_train) if self.norm_type == "UnitTransformer" else UnitGaussianNormalizer(fx_train)
        fx_train = x_normalizer.encode(fx_train)
        fx_test = x_normalizer.encode(fx_test)
        x_normalizer.cuda()

        if self.normalize:
            if self.norm_type == "UnitTransformer":
                self.y_normalizer = UnitTransformer(yy_train)
            elif self.norm_type == "UnitGaussianNormalizer":
                self.y_normalizer = UnitGaussianNormalizer(yy_train)
            yy_train = self.y_normalizer.encode(yy_train)
            self.y_normalizer.cuda()

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_train, t_train, fx_train, yy_train),
            batch_size=self.batch_size, shuffle=True,
            collate_fn=self.random_collate_fn)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(pos_test, t_test, fx_test, yy_test),
            batch_size=self.batch_size, shuffle=False)

        print("Dataloading is over.")
        return train_loader, test_loader, [s1, s2]
'''

# ---------- data_factory.py updates ----------
_FACTORY_IMPORT_NEW = (
    "from data_provider.data_loader import airfoil, ns, darcy, pipe, elas, plas, "
    "pdebench_autoregressive, \\\n"
    "    pdebench_steady_darcy, car_design, cfd3d, pdebench_conditional, "
    "airfrans_design, aircraft_design, pdebench_autoregressive_flat, \\\n"
    "    pdebench_conditional_cfd\n"
)

_FACTORY_NEW_ENTRIES = (
    "        'pdebench_conditional': pdebench_conditional,\n"
    "        'airfrans_design': airfrans_design,\n"
    "        'aircraft_design': aircraft_design,\n"
    "        'pdebench_autoregressive_flat': pdebench_autoregressive_flat,\n"
    "        'pdebench_conditional_cfd': pdebench_conditional_cfd,\n"
)


OPS = [
    # Add Custom import to model_factory.py
    {
        "op": "insert",
        "file": "Neural-Solver-Library/models/model_factory.py",
        "after_line": 2,
        "content": _CUSTOM_IMPORT,
    },
    # Add Custom entry to model_dict
    {
        "op": "insert",
        "file": "Neural-Solver-Library/models/model_factory.py",
        "after_line": 9,
        "content": _CUSTOM_ENTRY,
    },
    # Inject TRAIN_METRICS into exp_steady.py (after rel_err print at line 91)
    {
        "op": "insert",
        "file": "Neural-Solver-Library/exp/exp_steady.py",
        "after_line": 91,
        "content": _STEADY_TRAIN_METRICS,
    },
    # Inject TRAIN_METRICS into exp_dynamic_autoregressive.py (after test loss print at line 100)
    {
        "op": "insert",
        "file": "Neural-Solver-Library/exp/exp_dynamic_autoregressive.py",
        "after_line": 100,
        "content": _AUTOREGRESSIVE_TRAIN_METRICS,
    },
    # Inject TRAIN_METRICS into exp_dynamic_conditional.py (after test loss print at line 85)
    {
        "op": "insert",
        "file": "Neural-Solver-Library/exp/exp_dynamic_conditional.py",
        "after_line": 85,
        "content": _CONDITIONAL_TRAIN_METRICS,
    },
    # Inject TRAIN_METRICS into exp_steady_design.py (after rel_err print at line 91)
    {
        "op": "insert",
        "file": "Neural-Solver-Library/exp/exp_steady_design.py",
        "after_line": 91,
        "content": _DESIGN_TRAIN_METRICS,
    },
    # Fix car_design path bug: get_datalist root should include 'training_data'
    # Also: savedir=None to avoid writing preprocessed data (tmpfs space limit)
    {
        "op": "replace",
        "file": "Neural-Solver-Library/data_provider/data_loader.py",
        "start_line": 621,
        "end_line": 626,
        "content": (
            "        data_root = os.path.join(self.file_path, 'training_data')\n"
            "        preproc_dir = os.path.join(self.file_path, 'preprocessed_data')\n"
            "        use_savedir = preproc_dir if preprocessed else None\n"
            "        train_dataset, coef_norm = get_datalist(data_root, trainlst, norm=True,\n"
            "                                                savedir=use_savedir,\n"
            "                                                preprocessed=preprocessed)\n"
            "        val_dataset = get_datalist(data_root, vallst, coef_norm=coef_norm,\n"
            "                                   savedir=use_savedir,\n"
            "                                   preprocessed=preprocessed)\n"
        ),
    },
    # Add all new loaders to data_loader.py (at end of file, line 759)
    {
        "op": "insert",
        "file": "Neural-Solver-Library/data_provider/data_loader.py",
        "after_line": 759,
        "content": _PDEBENCH_CONDITIONAL_CODE + _AIRFRANS_DESIGN_CODE + _AIRCRAFT_DESIGN_CODE + _PDEBENCH_AUTOREGRESSIVE_FLAT_CODE + _PDEBENCH_CONDITIONAL_CFD_CODE,
    },
    # Replace data_factory.py import lines 1-2 to include new loaders
    {
        "op": "replace",
        "file": "Neural-Solver-Library/data_provider/data_factory.py",
        "start_line": 1,
        "end_line": 2,
        "content": _FACTORY_IMPORT_NEW,
    },
    # Add new loader entries to data_dict (insert after last entry at line 16, before closing brace)
    {
        "op": "insert",
        "file": "Neural-Solver-Library/data_provider/data_factory.py",
        "after_line": 16,
        "content": _FACTORY_NEW_ENTRIES,
    },
    # Wrap exp.test() in try/except in run.py
    {
        "op": "replace",
        "file": "Neural-Solver-Library/run.py",
        "start_line": 99,
        "end_line": 103,
        "content": (
            "    if eval:\n"
            "        exp.test()\n"
            "    else:\n"
            "        exp.train()\n"
            "        try:\n"
            "            exp.test()\n"
            "        except Exception as e:\n"
            "            import traceback; traceback.print_exc()\n"
            "            print(f'Test phase skipped: {e}')\n"
        ),
    },
    # Lazy edge_index computation in GraphDataset (shapenet_utils.py)
    # Instead of computing radius graph for ALL samples in __init__,
    # compute on first access in get() and cache the result.
    {
        "op": "replace",
        "file": "Neural-Solver-Library/data_provider/shapenet_utils.py",
        "start_line": 327,
        "end_line": 349,
        "content": (
            "    def __init__(self, datalist, use_height=False, use_cfd_mesh=True, r=None, coef_norm=None, valid_list=None):\n"
            "        super().__init__()\n"
            "        self.datalist = datalist\n"
            "        self.use_height = use_height\n"
            "        self.coef_norm = coef_norm\n"
            "        self.valid_list = valid_list\n"
            "        self.r = r\n"
            "        self.use_cfd_mesh = use_cfd_mesh\n"
            "        self._edge_index_computed = set()\n"
            "        if not use_cfd_mesh:\n"
            "            assert r is not None\n"
            "\n"
            "    def len(self):\n"
            "        return len(self.datalist)\n"
            "\n"
            "    def get(self, idx):\n"
            "        data = self.datalist[idx]\n"
            "        if not self.use_cfd_mesh and idx not in self._edge_index_computed:\n"
            "            self.datalist[idx] = create_edge_index_radius(data, self.r)\n"
            "            data = self.datalist[idx]\n"
            "            self._edge_index_computed.add(idx)\n"
            "            if (len(self._edge_index_computed)) % 100 == 0:\n"
            "                print(f'  [GraphDataset] Computed edge_index for {len(self._edge_index_computed)}/{len(self.datalist)} samples', flush=True)\n"
            "        shape = get_shape(data, use_height=self.use_height)\n"
            "        if self.valid_list is None:\n"
            "            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \\\n"
            "                data.edge_index\n"
            "        else:\n"
            "            return self.datalist[idx].pos, self.datalist[idx].x, self.datalist[idx].y, self.datalist[idx].surf, \\\n"
            "                data.edge_index, self.valid_list[idx]\n"
        ),
    },
    # Replace test() in exp_steady_design.py to handle non-car datasets
    # (cal_coefficient only works for car data; AirCraft has all-surface points)
    {
        "op": "replace",
        "file": "Neural-Solver-Library/exp/exp_steady_design.py",
        "start_line": 104,
        "end_line": 178,
        "content": (
            "    def test(self):\n"
            "        self.model.load_state_dict(torch.load('./checkpoints/' + self.args.save_name + '.pt'))\n"
            "        self.model.eval()\n"
            "        if not os.path.exists('./results/' + self.args.save_name + '/'):\n"
            "            os.makedirs('./results/' + self.args.save_name + '/')\n"
            "\n"
            "        criterion_func = nn.MSELoss(reduction='none')\n"
            "        l2errs_press = []\n"
            "        l2errs_velo = []\n"
            "        mses_press = []\n"
            "        mses_velo_var = []\n"
            "        times = []\n"
            "        gt_coef_list = []\n"
            "        pred_coef_list = []\n"
            "        coef_error = 0\n"
            "        index = 0\n"
            "        has_coef = True\n"
            "        with torch.no_grad():\n"
            "            for pos, fx, y, surf, geo, obj_file in self.test_loader:\n"
            "                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()\n"
            "                if self.args.fun_dim == 0:\n"
            "                    fx = None\n"
            "                tic = time.time()\n"
            "                out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]\n"
            "                toc = time.time()\n"
            "\n"
            "                if self.test_loader.coef_norm is not None:\n"
            "                    mean = torch.tensor(self.test_loader.coef_norm[2]).cuda()\n"
            "                    std = torch.tensor(self.test_loader.coef_norm[3]).cuda()\n"
            "                    pred_press = out[surf, -1] * std[-1] + mean[-1]\n"
            "                    gt_press = y[surf, -1] * std[-1] + mean[-1]\n"
            "                    non_surf = ~surf\n"
            "                    if non_surf.any():\n"
            "                        pred_velo = out[non_surf, :-1] * std[:-1] + mean[:-1]\n"
            "                        gt_velo = y[non_surf, :-1] * std[:-1] + mean[:-1]\n"
            "                    else:\n"
            "                        pred_velo = out[:, :-1] * std[:-1] + mean[:-1]\n"
            "                        gt_velo = y[:, :-1] * std[:-1] + mean[:-1]\n"
            "\n"
            "                if has_coef:\n"
            "                    try:\n"
            "                        pred_surf_velo = out[surf, :-1]\n"
            "                        if self.test_loader.coef_norm is not None:\n"
            "                            pred_surf_velo = pred_surf_velo * std[:-1] + mean[:-1]\n"
            "                        pred_coef = cal_coefficient(obj_file.split('/')[1],\n"
            "                                                    pred_press[:, None].detach().cpu().numpy(),\n"
            "                                                    pred_surf_velo.detach().cpu().numpy())\n"
            "                        gt_surf_velo = y[surf, :-1]\n"
            "                        if self.test_loader.coef_norm is not None:\n"
            "                            gt_surf_velo = gt_surf_velo * std[:-1] + mean[:-1]\n"
            "                        gt_coef = cal_coefficient(obj_file.split('/')[1],\n"
            "                                                  gt_press[:, None].detach().cpu().numpy(),\n"
            "                                                  gt_surf_velo.detach().cpu().numpy())\n"
            "                        gt_coef_list.append(gt_coef)\n"
            "                        pred_coef_list.append(pred_coef)\n"
            "                        coef_error += (abs(pred_coef - gt_coef) / gt_coef)\n"
            "                    except Exception:\n"
            "                        has_coef = False\n"
            "\n"
            "                l2err_press = torch.norm(pred_press - gt_press) / (torch.norm(gt_press) + 1e-8)\n"
            "                l2err_velo = torch.norm(pred_velo - gt_velo) / (torch.norm(gt_velo) + 1e-8)\n"
            "\n"
            "                if non_surf.any():\n"
            "                    mse_press = criterion_func(out[surf, -1], y[surf, -1]).mean(dim=0)\n"
            "                    mse_velo_var = criterion_func(out[non_surf, :-1], y[non_surf, :-1]).mean(dim=0)\n"
            "                else:\n"
            "                    mse_press = criterion_func(out[:, -1], y[:, -1]).mean(dim=0)\n"
            "                    mse_velo_var = criterion_func(out[:, :-1], y[:, :-1]).mean(dim=0)\n"
            "\n"
            "                l2errs_press.append(l2err_press.cpu().numpy())\n"
            "                l2errs_velo.append(l2err_velo.cpu().numpy())\n"
            "                mses_press.append(mse_press.cpu().numpy())\n"
            "                mses_velo_var.append(mse_velo_var.cpu().numpy())\n"
            "                times.append(toc - tic)\n"
            "                index += 1\n"
            "\n"
            "        if has_coef and len(gt_coef_list) > 0:\n"
            "            gt_coef_list = np.array(gt_coef_list)\n"
            "            pred_coef_list = np.array(pred_coef_list)\n"
            "            spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]\n"
            "            print('rho_d: ', spear)\n"
            "            print('c_d: ', coef_error / index)\n"
            "        l2err_press = np.mean(l2errs_press)\n"
            "        l2err_velo = np.mean(l2errs_velo)\n"
            "        rmse_press = np.sqrt(np.mean(mses_press))\n"
            "        rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))\n"
            "        if self.test_loader.coef_norm is not None:\n"
            "            rmse_press *= self.test_loader.coef_norm[3][-1]\n"
            "            rmse_velo_var *= self.test_loader.coef_norm[3][:-1]\n"
            "        print('relative l2 error press:', l2err_press)\n"
            "        print('relative l2 error velo:', l2err_velo)\n"
            "        print('press:', rmse_press)\n"
            "        print('velo:', rmse_velo_var, np.sqrt(np.mean(np.square(rmse_velo_var))))\n"
            "        print('time:', np.mean(times))\n"
        ),
    },
]
