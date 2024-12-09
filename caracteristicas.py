from sklearn.feature_selection import SelectKBest, f_classif
X_new = SelectKBest(f_classif, k=5).fit_transform(X, y)
