import numpy as np
from torch.utils.data import Dataset
from typing import List
import glob
from torchvision import transforms

def get_data_2d(task, sizes=np.arange(5, 16), data_augmentation=True, test_split=0.2, temp=1.0, add_parameters_as_channel=False, rescaled=False,
                rescale_output=False, data_fraction=1.0, ztU_range=None, only_converged=None, limit_energy=None, savedir_no_backup=False, checkerboard_channel=False,
                ignore_large_real_sweeps=False, limit_ztU_range_in_training_data=False, qmc_path=None, classifier_name=None):

    if temp == 0.25:
        task = task + '_temp_025'

    if qmc_path is None:
        if not savedir_no_backup:
            path = os.getcwd() + '/QMC/data/potentials/' + str(task)
        else:
            path = '/ptmp/bale/data/qmc/' + str(task)

    else:
        path = qmc_path + "/" + str(task)
        print(path)

    paths = glob.glob(path + "/*")

    if classifier_name is not None:
        if "decision_tree" in classifier_name:
            classifier = pickle.load(open(DATA_ROOT/f"models/2d/classifiers/{classifier_name}.p","rb"))

    # # rearrange using saved process number (make sure that with a new loading the new testset has not been trained on before)
    new_paths = [[] for i in range(10000)]
    for path in paths:
        idx = int(re.split("_|\\.", path)[-2])
        new_paths[idx] = path

    final_paths = []
    for path in new_paths:
        if path != []:
            final_paths.append(path)

    print('\nFound %d datasets' % len(final_paths))

    # load data
    # [ [] for i in range(np.min(sizes),np.max(sizes)+1) ]
    total_potentials = [[] for i in sizes]
    # [ [] for i in range(np.min(sizes),np.max(sizes)+1) ]
    total_labels = [[] for i in sizes]

    total_nr = 0
    warned_sizes = []
    ztU_notification = False
    not_converged = 0
    high_energy = 0

    for path in paths:

        data = pickle.load(open(path, "rb"),  encoding='latin1')

        for key in data:
            # [mu_2D_str, translated_mu_2D, density, density_error, energy, energy_error, qmc_particles, (particles,) thermalization, converged, duration, t, U, V, Nmax, offset, prefactors]
            # data_dict[(model, potential_description, t, U, V, offset, amplitude, sweeps, skips)]
            # = [mu_2D_str, translated_mu_2D, density, density_error, energy, energy_error, qmc_particles, thermalization, converged, duration, t, U, V, Nmax, T, sweeps, skips, tau]
            #try:
            size = np.array(data[key][1]).shape[0]
            idx = list(sizes).index(size)

            potential = data[key][1]
            converged = data[key][8]

            # ignore not converged samples if desired
            if not converged and only_converged:
                not_converged += 1
                continue

            # ignore high energy samples if desired (eliminates phase shifts)
            if limit_energy:
                if data[key][4] > limit_energy:
                    high_energy += 1
                    continue

            if abs(data[key][2]).mean()==0 or data[key][4]==0:
                continue

            if classifier_name is not None:
                if "decision_tree" in classifier_name:
                    features = get_decision_tree_classifier_features(data[key][2], data[key][3], data[key][4], data[key][5], data[key][1])

                    if not classifier.predict(features)[0]==1:
                        continue


            if ignore_large_real_sweeps:
                if data[key][-1] > 93:
                    continue

            # add checkerboard suggestion to deal with symmetry breaking
            if checkerboard_channel:

                debug_mode = False

                # extend axis
                potential = potential[:, :, np.newaxis]

                # create checkerboards
                cb1 = np.zeros(potential.shape)
                cb1[::2, ::2] = 1
                cb1[1::2, 1::2] = 1

                cb2 = 1 - cb1

                if debug_mode:
                    print(cb1.shape)
                    plt.imshow(cb1[:, :, 0])
                    plt.show()
                    print(cb2.shape)
                    plt.imshow(cb2[:, :, 0])
                    plt.show()

                # projections
                density = data[key][2][:, :, np.newaxis]

                if debug_mode:
                    print(density.shape)
                    plt.imshow(density[:, :, 0])
                    plt.show()

                p1 = np.sum(cb1*density)
                p2 = np.sum(cb2*density)

                if debug_mode:
                    print(p1, p2)
                    print('end')

                # add checkerboard with bigger projection
                if p1 > p2:
                    potential = np.concatenate((potential, cb1), axis=2)
                else:
                    potential = np.concatenate((potential, cb2), axis=2)

            if not add_parameters_as_channel:
                full_data_single_sample = potential
            else:
                # extend dimension
                if len(potential.shape) == 2:
                    potential = potential[:, :, np.newaxis]

                # add hamilton parameters
                U = np.zeros((potential.shape[0], potential.shape[1], 1))
                U.fill(float(key[3]))

                V = np.zeros((potential.shape[0], potential.shape[1], 1))
                V.fill(float(key[4]))

                if rescaled:
                    if ztU_range and not ztU_notification:
                        print('restricting ztU range to', ztU_range)
                        ztU_notification = True

                    # use rescaled ratios from paper rather than plain values
                    muU = potential/U  # mu/U is ca - 1 to 4
                    muU = (muU + 1)/5  # rescale to about [0,1]

                    ztU = 4 * 1 / U  # z * t / U   is [0,0.6]
                    ztU = ztU / 0.6

                    zVU = 4 * V / U  # z * V / U is [0.75,1.75]
                    zVU = zVU - 0.75

                    # check if in valid ztU range
                    if ztU_range:
                        if ztU[0][0][0]*0.6 >= ztU_range[0] and ztU[0][0][0]*0.6 <= ztU_range[1]:
                            full_data_single_sample = np.concatenate(
                                (muU, ztU, zVU), axis=2)
                        else:
                            continue
                    else:
                        full_data_single_sample = np.concatenate(
                            (muU, ztU, zVU), axis=2)
                else:
                    if ztU_range and not ztU_notification:
                        print(
                            'ztU range restriction currently not supported without rescaled parameters')
                        ztU_notification = True

                    full_data_single_sample = np.concatenate(
                        (potential, U, V), axis=2)

            total_potentials[idx].append(full_data_single_sample)

            if not rescale_output:
                total_labels[idx].append(data[key][2])
            else:
                total_labels[idx].append(data[key][2]/3)

            total_nr += 1
            #except:
                #size = np.array(data[key][1]).shape[0]

                #if size not in warned_sizes:
                #    print('ignored size', size, ' with key ', key)
                #    warned_sizes.append(size)

    print('\nnot converged samples which were ignored (only nonzero if \'only_converged\' == True)', not_converged)
    print('\nhigh energy samples which were ignored (to prevent phase shift)', high_energy)

    print('\ndata per system size:')

    for i, labels in enumerate(total_labels):
        print(sizes[i], len(labels))

    # get minimum of samples per gridsize accross gridsizes
    for i in range(len(total_potentials)):
        if i == 0:
            minimum = len(total_potentials[i])
        else:
            if len(total_potentials[i]) < minimum:
                minimum = len(total_potentials[i])

    print('total dataset size is %d' % (total_nr))
    print('\nresizing dataset to get the same amount of examples per system size')
    print('total dataset size will be %d' % (len(sizes)*minimum))

    # split into train and test data + make sure all system sizes have the same amount of data

    x_train = []
    x_test = []
    y_train = []
    y_test = []

    for i in range(len(total_potentials)):
        # use either 'len(total_potentials[i])' (full number of training samples) or 'minimum' (reduces number of training samples)
        cut = int(data_fraction*minimum)

        x_train.append(total_potentials[i][:int((1-test_split)*cut)])
        x_test.append(total_potentials[i][int((1-test_split)*cut):cut])

        y_train.append(total_labels[i][:int((1-test_split)*cut)])
        y_test.append(total_labels[i][int((1-test_split)*cut):cut])

    # filter training data ztU range
    if limit_ztU_range_in_training_data and ztU_range:
        print('\nWARNING: Filtering ztU range in training data\n')

        x_train_buffer = [[] for i in range(len(x_train))]
        y_train_buffer = [[] for i in range(len(y_train))]

        # gridsize
        for i in range(len(x_train)):
            # samples
            for j in range(len(x_train[i])):

                ztU = 4/x_train[i][j][1][0, 0]
                if ztU < ztU_range[0] or ztU > ztU_range[1]:
                    continue
                else:
                    x_train_buffer[i].append(x_train[i][j])
                    y_train_buffer[i].append(y_train[i][j])

        # make sure they are all of the same size
        # get minimum
        minimum = len(x_train_buffer[0])
        for i in range(len(x_train_buffer)):
            if len(x_train_buffer[i]) < minimum:
                minimum = len(x_train_buffer)

        # cut off at minimum
        for i in range(len(x_train_buffer)):
            x_train_buffer[i] = x_train_buffer[i][:minimum]
            y_train_buffer[i] = y_train_buffer[i][:minimum]

        x_train = x_train_buffer
        y_train = y_train_buffer

    if data_augmentation:

        print('\nApplying data augmentation on trainset of size',
              len(x_train)*len(x_train[0]), '...\n')

        # Augment all training data - not speed optimized
        x_train_buffer = [[] for i in range(len(x_train))]
        y_train_buffer = [[] for i in range(len(y_train))]

        # gridsize
        for i in range(len(x_train)):
            # samples
            for j in range(len(x_train[i])):
                x_train_buffer[i].append(copy.deepcopy(x_train[i][j]))
                y_train_buffer[i].append(copy.deepcopy(y_train[i][j]))

                x_train_buffer[i].append(
                    rotate(copy.deepcopy(x_train[i][j]), 1))
                y_train_buffer[i].append(
                    rotate(copy.deepcopy(y_train[i][j]), 1))

                x_train_buffer[i].append(
                    rotate(copy.deepcopy(x_train[i][j]), 2))
                y_train_buffer[i].append(
                    rotate(copy.deepcopy(y_train[i][j]), 2))

                x_train_buffer[i].append(
                    rotate(copy.deepcopy(x_train[i][j]), 3))
                y_train_buffer[i].append(
                    rotate(copy.deepcopy(y_train[i][j]), 3))

                x_train_buffer[i].append(
                    mirror_x(copy.deepcopy(x_train[i][j])))
                y_train_buffer[i].append(
                    mirror_x(copy.deepcopy(y_train[i][j])))
                x_train_buffer[i].append(
                    mirror_y(copy.deepcopy(x_train[i][j])))
                y_train_buffer[i].append(
                    mirror_y(copy.deepcopy(y_train[i][j])))

                x_train_buffer[i].append(
                    mirror_x(rotate(copy.deepcopy(x_train[i][j]), 1)))
                y_train_buffer[i].append(
                    mirror_x(rotate(copy.deepcopy(y_train[i][j]), 1)))
                x_train_buffer[i].append(
                    mirror_y(rotate(copy.deepcopy(x_train[i][j]), 1)))
                y_train_buffer[i].append(
                    mirror_y(rotate(copy.deepcopy(y_train[i][j]), 1)))

            print_progress(i+1, len(x_train),
                           prefix='Progress:', suffix='Complete')

        x_train = x_train_buffer
        y_train = y_train_buffer

        print('\nTotal trainset size is now %d' %
              (len(x_train)*len(x_train[0])))

    return x_train, y_train, x_test, y_test


class DataSet2d(Dataset):
    def __init__(self, task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup=True, checkerboard_channel=True, ignore_large_real_sweeps=True, augmentations=True, classifier_name=None, stage="train",**kwargs) -> None:
        super().__init__()

        self.task = task
        self.sizes = sizes
        self.add_parameters_as_channel = add_parameters_as_channel
        self.rescaled = rescaled
        self.rescale_output = rescale_output
        self.data_fraction = data_fraction
        self.test_split = test_split
        self.ztU_range = ztU_range
        self.only_converged = only_converged
        self.limit_energy = limit_energy
        self.savedir_no_backup = savedir_no_backup
        self.checkerboard_channel = checkerboard_channel
        self.ignore_large_real_sweeps = ignore_large_real_sweeps
        self.classifier_name = classifier_name

        x_train, y_train, x_test, y_test = get_data_2d(task, sizes=sizes, data_augmentation=False, test_split=test_split,
                                                       add_parameters_as_channel=add_parameters_as_channel, rescaled=rescaled,
                                                       rescale_output=rescale_output, data_fraction=data_fraction, ztU_range=ztU_range,
                                                       only_converged=only_converged, limit_energy=limit_energy, savedir_no_backup=savedir_no_backup,
                                                       checkerboard_channel=checkerboard_channel, ignore_large_real_sweeps=ignore_large_real_sweeps, classifier_name=classifier_name)

        # transpose for pytorch
        x_train, y_train, x_test, y_test = map(lambda set: [
                                               [xy.T for xy in size_set] for size_set in set], (x_train, y_train, x_test, y_test))

        assert len(set(list(map(len, (x_train, y_train, x_test, y_test))))
                   ) == 1, "Sets ahave different amount of sizes"

        self.lengths_all_sets = np.array(list(map(lambda sets: [len(
            set_) for set_ in sets], zip(x_train, y_train, x_test, y_test))))
        assert np.unique(self.lengths_all_sets,
                         axis=0).shape[0] == 1, "sets have different lengths"

        self.augmentations = augmentations
        self.augmentation_transforms = transforms.Compose([
            RandomRotation(), RandomFlip(axis=-2), RandomFlip(axis=-1)
        ])

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

        self._stage = stage

    @property
    def stage(self):
        if not self._stage in ("test", "train"):
            raise RuntimeError("unknown stage")
        return self._stage

    @stage.setter
    def stage(self, v):
        self._stage = v

    @property
    def x_set(self):
        if self.stage == "train":
            return self.x_train

        elif self.stage == "test":
            return self.x_test

    @property
    def y_set(self):
        if self.stage == "train":
            return self.y_train

        elif self.stage == "test":
            return self.y_test

    @property
    def set_sizes(self):
        if self.stage == "train":
            return self.lengths_all_sets[:, 0]

        elif self.stage == "test":
            return self.lengths_all_sets[:, 2]

    @property
    def cumulative_sizes(self):
        return np.cumsum(self.set_sizes)

    @classmethod
    def loader(cls, task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup=True, checkerboard_channel=True, ignore_large_real_sweeps=True, classifier_name=None, augmentations=True, stage="train", create_new=False,**kwargs):

        if not create_new and cls.save_path(task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, classifier_name).is_file():

            loaded_dataset = pickle.load(open(cls.save_path(
                task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, classifier_name), "rb"))

            loaded_dataset.augmentations = augmentations
            loaded_dataset.stage = stage

            logger.info("Using stored dataset.")
            return loaded_dataset
        else:
            name = cls.save_path(
                task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, classifier_name).name
            logger.info(f"Creating new dataset {name}")

            obj = cls(task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range,
                      only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, augmentations, classifier_name, stage)

            obj.save()
            return obj

    @staticmethod
    def file_name(task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, classifier_name):
        return "_".join(map(str, [task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, classifier_name]))

    @classmethod
    def save_path(cls, task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps, classifier_name):
        return DATA_ROOT/f"models/2d/datasets/{cls.file_name(task, sizes, add_parameters_as_channel, rescaled, rescale_output, data_fraction, test_split, ztU_range, only_converged, limit_energy, savedir_no_backup, checkerboard_channel, ignore_large_real_sweeps,classifier_name)}.p"

    @property
    def _save_path(self):
        return self.save_path(self.task,
                              self.sizes,
                              self.add_parameters_as_channel,
                              self.rescaled,
                              self.rescale_output,
                              self.data_fraction,
                              self.test_split,
                              self.ztU_range,
                              self.only_converged,
                              self.limit_energy,
                              self.savedir_no_backup,
                              self.checkerboard_channel,
                              self.ignore_large_real_sweeps,
                              self.classifier_name)

    def save(self):
        pickle.dump(self, open(self._save_path, "wb"))

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):

        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]

        x, y = self.x_set[dataset_idx][sample_idx], self.y_set[dataset_idx][sample_idx]

        if self.augmentations:
            x, y = self.augmentation_transforms((x, y))

        return x, y

    def split_wrt_sizes(self, splits: List[float], split_version=0):

        splits_dir_path = self._save_path.parent / \
            f"splits/{self._save_path.name}"
        splits_dir_path.mkdir(exist_ok=True, parents=True)

        # search if split exists
        split_dict = {"splits":splits,"versions":{}}
        split_idx = sorted([-1] + list(map(lambda p: int(p.stem.split("_")
                           [-1]), splits_dir_path.glob("split_*"))))[-1] + 1
        splits_path = splits_dir_path/f"split_{split_idx}.json"

        for _split_path in splits_dir_path.glob("split_*"):
            with open(_split_path, "r") as file:
                _split_dict = json.load(file)

            if _split_dict.get("splits", None) is None:
                continue

            if not (len(_split_dict.get("splits", None)) == len(splits)):
                continue
            
            if np.allclose(np.array(_split_dict.get("splits", None)), np.array(splits)):
                
                if str(split_version) in map(str,_split_dict["versions"].keys()):
                    logger.info("Found existing Split")
                    split_indices = _split_dict["versions"][str(split_version)]
                    return split_indices
                else:
                    splits_path = _split_path
                    split_dict["versions"].update(_split_dict["versions"])
                    break


        if all(s>=1 for s in splits):
            logger.info("Calculating Fractions from integers")
            if not abs(int(sum(splits)) - len(self))<2:
                logger.info(f"Sum of integers {int(sum(splits))} not equal to set length {len(self)} -> add last split")
                splits.append(len(self) - int(sum(splits)))

            splits = [float(s)/len(self) for s in splits]


        logger.info(f"Creating new split for splits {splits}; Version {split_version}")

        np.random.seed(seed=split_version)
        assert np.allclose(sum(splits), 1)

        split_indices_all_sets = []

        for set_idx in range(len(self.x_set)):
            split_indices_all_sets.append([])

            offset = ([0]+list(self.cumulative_sizes))[set_idx]
            set_indices = np.arange(len(self.x_set[set_idx])) + offset
            set_length = len(set_indices)

            for split_fraction in splits[:-1]:

                split_indices_set = np.random.choice(set_indices, int(
                    split_fraction*set_length), replace=False)
                set_indices = np.array(
                    list(set(set_indices)-set(split_indices_set)))

                split_indices_all_sets[set_idx].append(split_indices_set)

            split_indices_all_sets[set_idx].append(set_indices)

        split_indices = list(map(lambda ind: [int(k) for k in np.concatenate(ind)], zip(*split_indices_all_sets)))

        # assert correct splitting
        assert sum(len(i) for i in split_indices) == len(self)
        assert len(set.intersection(*map(set, split_indices))) == 0
        assert len(set.union(*map(set, split_indices)) -
                   set(range(len(self)))) == 0

        split_dict["versions"].update({str(split_version):split_indices})

        with open(splits_path,"w") as file:
            json.dump(split_dict,file)

        return split_indices


def collate_sizes_2d(batch):

    sizes = np.array(tuple(map(lambda sample: sample[0].shape[-1], batch)))

    size_batches_in = []
    size_batches_out = []
    for size in set(sizes):

        size_batch_in, size_batch_out = map(lambda array: torch.from_numpy(np.stack(array)).float(
        ), zip(*[batch[sample_idx] for sample_idx in np.argwhere(sizes == size).flatten()]))

        size_batches_in.append(size_batch_in)
        size_batches_out.append(size_batch_out)

    return size_batches_in, size_batches_out