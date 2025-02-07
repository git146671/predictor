package com.pp.predictor;

import jakarta.annotation.PostConstruct;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Component;

import java.io.File;
import java.io.IOException;

@Component
public class CryptoModelTrainer {
    @PostConstruct
    public void init() throws IOException, InterruptedException {

        int numInputs = 1;
        int numOutputs = 1;
        int numHiddenNodes = 50;
        int batchSize = 32;
        int epochs = 50;

        // Загрузка данных
        CSVRecordReader csvReader = new CSVRecordReader(1, ',');
        csvReader.initialize(new FileSplit(new File("csv/BTCUSD.csv")));
        DataSetIterator iterator = new RecordReaderDataSetIterator(csvReader, batchSize, 1, 1, true);

        // Создание модели
        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.01))
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build())
                .build()
        );
        model.init();

        // Обучение модели
        for (int i = 0; i < epochs; i++) {
            model.fit(iterator);
            iterator.reset();
        }

        // Сохранение модели
        model.save(new File("model/crypto_model.zip"));
    }
}