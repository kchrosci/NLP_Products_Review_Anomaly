import train_model as t


def main():
    normal_opinions, anomaly_opinions = t.load_opinions()
    normal_opinions, normal_opinions_test = t.split_data(normal_opinions, anomaly_opinions)
    normal_dataset, normal_dataset_test, anomaly_dataset = t.create_tensors(normal_opinions, anomaly_opinions, normal_opinions_test)
    normal_dataset, normal_dataset_test, anomaly_dataset = t.align_tensors_shape(normal_dataset, normal_dataset_test, anomaly_dataset)
    autoencoder, threshold = t.start_training(normal_dataset)
    t.save_model(autoencoder, threshold)
    cm1 = t.test_normal_opinions(autoencoder, normal_dataset_test, threshold)
    cm2 = t.test_anomaly_opinions(autoencoder, anomaly_dataset, threshold)
    t.show_results(cm1, cm2)


if __name__ == "__main__":
    main()
