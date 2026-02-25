from abc import ABC, abstractmethod

import numpy as np


class MetricsComputer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def compute_cls_score(cls_stack, cls_token):
        pass

    @abstractmethod
    def compute_patch_scores(patch_stack, target_image_size, patch_tokens):
        pass


class MahalanobisMetric(MetricsComputer):
    @staticmethod
    def mean_cov_inv(features, eps=1e-2):
        """
        features: (N, D)
        """
        mean = np.mean(features, axis=0)
        diffs = features - mean
        cov = diffs.T @ diffs / (features.shape[0] - 1)
        cov += eps * np.eye(cov.shape[0], dtype=np.float32)
        cov_inv = np.linalg.inv(cov)
        return mean, cov_inv

    @staticmethod
    def mahalanobis_distance(vec, mean, cov_inv):
        """
        vec: (N, D)
        """
        diff = vec - mean
        d_sq = np.sum((diff @ cov_inv) * diff, axis=1)
        return np.sqrt(d_sq)

    @staticmethod
    def compute_cls_score(cls_stack, cls_token):
        mean_cls, covinv_cls = MahalanobisMetric.mean_cov_inv(cls_stack)
        cls_score = MahalanobisMetric.mahalanobis_distance(
            cls_token[None, :], mean_cls, covinv_cls
        )[0]
        return cls_score

    @staticmethod
    def compute_patch_scores(patch_stack, target_image_size, patch_tokens):
        mean_patch, covinv_patch = MahalanobisMetric.mean_cov_inv(patch_stack)
        num_real_patches = (target_image_size // 16) ** 2

        patch_tokens = patch_tokens[-num_real_patches:]
        diff = patch_tokens - mean_patch
        patch_scores = np.einsum("nd,df,nf->n", diff, covinv_patch, diff)
        return patch_scores
