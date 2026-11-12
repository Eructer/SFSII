class FilamentDataset(Dataset):
    def __init__(self, pairs, tile=512, crops_per_image=16, fg_prob=0.5, augment=False):
        """
        pairs: list of (image_path, mask_path)
        tile:  crop size; None returns the full image (validation/test)
        """
        self.pairs = list(pairs)
        self.tile = tile
        self.crops_per_image = crops_per_image if tile else 1
        self.fg_prob = fg_prob
        self.augment = augment

    def __len__(self):
        return len(self.pairs) * self.crops_per_image

    def __getitem__(self, index):
        img_path, mask_path = self.pairs[index // self.crops_per_image]
        img = np.array(Image.open(img_path).convert("L"))
        mask = (np.array(Image.open(mask_path)) > 0).astype(np.int64)

        if self.tile:
            H, W = mask.shape
            t = self.tile
            y = x = None
            if random.random() < self.fg_prob:
                fg = np.flatnonzero(mask)
                if fg.size:
                    cy, cx = divmod(int(fg[random.randrange(fg.size)]), W)
                    y = min(max(cy - t // 2, 0), H - t)
                    x = min(max(cx - t // 2, 0), W - t)
            if y is None:
                y, x = random.randint(0, H - t), random.randint(0, W - t)
            img = img[y:y + t, x:x + t]
            mask = mask[y:y + t, x:x + t]

        if self.augment:
            if random.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if random.random() < 0.5:
                img, mask = img[::-1], mask[::-1]
            k = random.randint(0, 3)
            img, mask = np.rot90(img, k), np.rot90(mask, k)

        image = torch.from_numpy(np.ascontiguousarray(img)).float().div(255.0).unsqueeze(0)
        return image, torch.from_numpy(np.ascontiguousarray(mask))