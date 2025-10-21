import os
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from scipy.stats import pearsonr
from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from joblib import Parallel, delayed
import numpy as np
from typing import Tuple
from scipy.stats import pearsonr

def pearson_corr(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    # print(y_pred)
    return pearsonr(y_true, y_pred)[0] 

def plot_alpha_dist(alphas: np.ndarray, file_path: str) -> None:
    """
    Plots the kernel density estimate (KDE) of the alpha distributions per electrode 
    and saves the plot to the specified file path.
    """
    n_electrodes = alphas.shape[1]
    colors = sns.color_palette("husl", n_electrodes)
    plt.figure(figsize=(10, 6))
    for i in range(n_electrodes):
        sns.kdeplot(alphas[:, i], color=colors[i], label=f'Electrode {i+1}', linewidth=2)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Alpha Distributions per Electrode")
    plt.legend()
    plt.savefig(file_path)


class RidgeRegression:
    def __init__(self, n_jobs, scoring: str = 'pearson', 
                 alphas: list = [10**i for i in range(-10, 10)]) -> None:
        self.scoring = scoring
        self.alphas = alphas
        self.n_jobs=n_jobs

    def get_scorer(self) -> callable:
        """Returns a scorer function based on the selected scoring metric."""
        if self.scoring == 'pearson':
            return make_scorer(lambda y, y_pred: pearson_corr(y, y_pred), greater_is_better=True)
        raise ValueError(f"Scoring method '{self.scoring}' not supported.")

    def select_best_alphas(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Select the best alpha for each electrode using RidgeCV on each time point.
        For each time point, pass the full multi-target y to RidgeCV, and aggregate 
        across time by computing the mode for each electrode.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features, n_time) or (n_samples, n_features)
            Feature data.
        y : np.ndarray, shape (n_samples, n_electrodes, n_time)
            Target data.
            
        Returns
        -------
        final_alphas : np.ndarray, shape (n_electrodes,)
            One selected alpha per electrode (the mode across time).
        alphas_time : np.ndarray, shape (n_time, n_electrodes)
            The best alpha from CV for each time point and electrode.
        """
        print("Finding optimal penalty terms...")

        # New Case 3: X and y are both 2D (no time dimension)
        if X.ndim == 2 and y.ndim == 2:
            # Here, y is of shape (n_samples, n_electrodes)
            n_electrodes = y.shape[1]
            # Run one multi-target ridge regression for all electrodes at once.
            # X = normalize(X)
            ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
                                scoring=self.get_scorer())
            ridge_cv.fit(X, y)
            final_alphas = ridge_cv.alpha_  # shape: (n_electrodes,)
            # Create a dummy time dimension (1, n_electrodes)
            alphas_time = final_alphas.reshape(1, -1)
    
        elif X.ndim == 2 and y.ndim == 3:
            _, n_electrodes, n_time = y.shape
            alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
            scores = np.zeros((n_time, n_electrodes), dtype=np.float32)
            
            # X = normalize(X)
            # Define helper for a single time slice.
            def process_time_slice_2d(t: int) -> Tuple[np.ndarray, np.ndarray]:
                y_t = y[:, :, t]  # shape: (n_samples, n_electrodes)
                
                ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
                                scoring=self.get_scorer())
                ridge_cv.fit(X, y_t)
                return ridge_cv.alpha_, ridge_cv.best_score_
            
            # Parallelize over time slices.
            results = Parallel(n_jobs=self.n_jobs, verbose=5)(
                delayed(process_time_slice_2d)(t) for t in range(n_time)
            )
            for t, (alpha_vals, score_vals) in enumerate(results):
                alphas_time[t, :] = alpha_vals
                scores[t, :] = score_vals
            
            final_alphas = np.zeros(n_electrodes, dtype=np.float32)
            for e in range(n_electrodes):
                final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
            print('Alpha values:', final_alphas)
            
        elif X.ndim == 3 and y.ndim == 3:

            _, n_electrodes, n_time = y.shape
            alphas_time = np.zeros((n_time, n_electrodes), dtype=np.float32)
            
            # Define helper for a single time slice in the 3D case.
            def process_time_slice_3d(t: int) -> np.ndarray:
                X_t = X[:, :, t]  # shape: (n_samples, n_features)
                y_t = y[:, :, t]  # shape: (n_samples, n_electrodes)
                # X_t = normalize(X_t)
                ridge_cv = RidgeCV(alphas=self.alphas, alpha_per_target=True, fit_intercept=True, 
                                scoring=self.get_scorer())
                ridge_cv.fit(X_t, y_t)
                return ridge_cv.alpha_
            
            results = Parallel(n_jobs=self.n_jobs, verbose=5)(
                delayed(process_time_slice_3d)(t) for t in range(n_time)
            )
            for t, alpha_vals in enumerate(results):
                alphas_time[t, :] = alpha_vals
            
            final_alphas = np.zeros(n_electrodes, dtype=np.float32)
            for e in range(n_electrodes):
                final_alphas[e] = np.atleast_1d(mode(alphas_time[:, e]).mode)[0]
            print('Alpha values:', final_alphas)
            
        else:
            raise ValueError("X must be a 2D or 3D array.")
        
        return final_alphas, alphas_time

    def cv(self, X: np.ndarray, y: np.ndarray, parallelize_choice) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs 5-fold cross-validation on the dataset, returning predictions, true values, 
        best alpha values, and beta coefficients for each fold.
        
        Parameters
        ----------
        X : np.ndarray
            Feature data. Either shape (n_samples, n_features) or (n_samples, n_features, n_time).
        y : np.ndarray
            Target data. Either shape (n_samples, n_electrodes) or (n_samples, n_electrodes, n_time).
            
        Returns
        -------
        y_pred_full : np.ndarray
            Predictions with shape (n_samples, n_electrodes, n_time).
        y_true_full : np.ndarray
            True targets with shape (n_samples, n_electrodes, n_time).
        alphas_time : np.ndarray
            The best alpha values obtained (with a dummy time dimension if data are 2D).
        betas : np.ndarray
            The beta coefficients for each fold, averaged over folds, with shape [n_electrodes, n_features, n_time].
        """
        kf = KFold(n_splits=5, shuffle=False)
        n_samples = X.shape[0]
        
        # Determine dimensions from y.
        if y.ndim == 2:
            n_time = 1
            n_electrodes = y.shape[1]
            y = y.reshape(y.shape[0], y.shape[1], 1)
        elif y.ndim == 3:
            n_time = y.shape[2]
            n_electrodes = y.shape[1]
        else:
            raise ValueError("y must be either 2D or 3D.")

        # Preallocate outputs
        y_pred_full = np.zeros((n_samples, n_electrodes, n_time), dtype=np.float32)
        y_true_full = np.zeros((n_samples, n_electrodes, n_time), dtype=y.dtype)
        betas = np.zeros((n_electrodes, X.shape[1], n_time), dtype=np.float32)  # To store betas for each fold

        def process_time_slice(t: int, X_train_fold, X_test_fold, y_train_fold, best_alphas, ndim: int) -> Tuple[np.ndarray, np.ndarray]:
            if ndim == 2:
                X_train_t = X_train_fold
                X_test_t = X_test_fold
                y_train_t = y_train_fold[:, :, t]
            else:
                X_train_t = X_train_fold[:, :, t]
                X_test_t = X_test_fold[:, :, t]
                y_train_t = y_train_fold[:, :, t]
            
            # X_train_t, X_test_t = normalize(X_train_t, X_test_t) 
            ridge = Ridge(alpha=best_alphas, fit_intercept=True)
            ridge.fit(X_train_t, y_train_t)
            return ridge.predict(X_test_t), ridge.coef_  # Return both prediction and betas

        def process_fold(fold, train_idx, test_idx):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_test_fold = X[test_idx]
            y_test_fold = y[test_idx]
            best_alphas, alphas_time = self.select_best_alphas(X_train_fold, y_train_fold)
            
            fold_pred = np.zeros((len(test_idx), n_electrodes, n_time), dtype=np.float32)
            fold_betas = np.zeros((n_electrodes, X.shape[1], n_time), dtype=np.float32)  # Store betas for this fold

            for t in range(n_time):
                if X.ndim == 2:
                    pred_t, betas_t = process_time_slice(t, X_train_fold, X_test_fold, y_train_fold, best_alphas, ndim=2)
                else:
                    pred_t, betas_t = process_time_slice(t, X_train_fold, X_test_fold, y_train_fold, best_alphas, ndim=3)

                fold_pred[:, :, t] = pred_t
                fold_betas[:, :, t] = betas_t  # Directly store betas for each fold and time slice

            return test_idx, fold_pred, y_test_fold, alphas_time, fold_betas

        # Now choose the parallelization strategy.
        if parallelize_choice == "folds":
            # Parallelize over folds.
            fold_results = Parallel(n_jobs=self.n_jobs)(
                delayed(process_fold)(fold, train_idx, test_idx)
                for fold, (train_idx, test_idx) in enumerate(kf.split(X))
            )
            for test_idx, fold_pred, y_test_fold, alphas_time, fold_betas in fold_results:
                y_pred_full[test_idx, :, :] = fold_pred
                y_true_full[test_idx, :, :] = y_test_fold
                betas += fold_betas  # Aggregate betas for each fold

        elif parallelize_choice == "time":
            # Sequentially process folds, but parallelize over time slices within each fold.
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                print(f"Fold {fold+1}: Training on {len(train_idx)} samples, testing on {len(test_idx)} samples.")
                X_train_fold = X[train_idx]
                y_train_fold = y[train_idx]
                X_test_fold = X[test_idx]
                y_test_fold = y[test_idx]
                best_alphas, alphas_time = self.select_best_alphas(X_train_fold, y_train_fold)
                print("Applying best alphas...")
                fold_pred = np.zeros((len(test_idx), n_electrodes, n_time), dtype=np.float32)
                
                # Parallelize over time slices
                results = Parallel(n_jobs=self.n_jobs, verbose=5)(
                    delayed(process_time_slice)(t, X_train_fold, X_test_fold, y_train_fold, best_alphas, X.ndim)
                    for t in range(n_time)
                )
                
                # Process the predictions and betas together
                for t, (pred, betas_t) in enumerate(results):
                    fold_pred[:, :, t] = pred
                    betas[:, :, t] += betas_t  # Aggregate betas for each fold

                y_pred_full[test_idx, :, :] = fold_pred
                y_true_full[test_idx, :, :] = y_test_fold

        else:
            raise ValueError("Please specify 'folds' or 'time' for parallelization.")

        return y_pred_full, y_true_full


