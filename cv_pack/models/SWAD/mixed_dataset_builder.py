import os
import random
import shutil


class MixedDatasetBuilder:
    """
    Costruisce una cartella 'mixed' a partire da dataset_path che contiene:
      - /good
      - /bad
    Inserisce le prime N immagini good (in ordine casuale), poi mescola
    tutte le altre good con le bad. Copia i file rinominandoli con indice
    incrementale e salva gli indici dei bad in bad_indices.txt.
    """

    def __init__(self, dataset_path, first_n_good=20, seed=None):
        self.dataset_path = dataset_path
        self.good_dir = os.path.join(dataset_path, "good")
        self.bad_dir = os.path.join(dataset_path, "bad")
        self.mixed_dir = os.path.join(dataset_path, "mixed")
        self.first_n_good = first_n_good

        if seed is not None:
            random.seed(seed)

        os.makedirs(self.mixed_dir, exist_ok=True)

    def _list_files(self, folder):
        return [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

    def build(self):
        # --- Carica file ---
        good_files = self._list_files(self.good_dir)
        bad_files = self._list_files(self.bad_dir)

        # --- Shuffle good ---
        random.shuffle(good_files)
        first_n_good = good_files[: self.first_n_good]
        remaining_good = good_files[self.first_n_good :]

        # --- Mescola resto good + bad ---
        mixed_pool = [(f, "good") for f in remaining_good] + [(f, "bad") for f in bad_files]
        random.shuffle(mixed_pool)

        # --- Ordine finale ---
        ordered_files = [(f, "good") for f in first_n_good] + mixed_pool

        # --- Copia e rinomina ---
        bad_indices = []
        for idx, (filepath, label) in enumerate(ordered_files, start=1):
            ext = os.path.splitext(filepath)[1]
            new_filename = f"{idx}{ext}"
            shutil.copy(filepath, os.path.join(self.mixed_dir, new_filename))
            if label == "bad":
                bad_indices.append(idx)

        # --- Salva indici bad ---
        bad_index_file = os.path.join(self.dataset_path, "bad_indices.txt")
        with open(bad_index_file, "w") as f:
            f.writelines(f"{index}\n" for index in bad_indices)

        # --- Log ---

        return {
            "total_mixed": len(ordered_files),
            "total_bad": len(bad_indices),
            "bad_indices": bad_indices,
            "bad_index_file": bad_index_file,
        }
