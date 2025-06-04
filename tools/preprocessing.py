
features = data.drop(['label'], axis=1)

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)


# TODO: PCA pour clustering 