package biz.k11i.xgboost;


import ml.dmlc.xgboost4j.java.Booster;
import ml.dmlc.xgboost4j.java.XGBoost;
import ml.dmlc.xgboost4j.java.DMatrix;
import ml.dmlc.xgboost4j.java.XGBoostError;

import java.io.IOException;
import java.util.*;


public class TestData {
    static final int NUM_DIMENSIONS = 100000;
    static final String MODEL_PATH = "model/test-generated.model";

    static float[] generateRow(Random r, int ncol, int label) {
        double b = ((double) label - 0.5) / 3.0;
        float[] result = new float[ncol];

        for (int i = 0; i < ncol; ++i) {
            double val = r.nextGaussian() * 2.0;
            if (i % 100 == 0) {
                val += b * r.nextDouble();
            }

            result[i] = (float) val;
        }

        return result;
    }

    static DMatrix generateDMatrix(int seed, int nrow, int ncol) throws XGBoostError {
        float[] matrix = new float[nrow * ncol];
        float[] labels = new float[nrow];
        Random r = new Random((long) seed);

        for (int i = 0; i < nrow; ++i) {
            int label = i % 2;
            labels[i] = (float) label;

            float[] row = generateRow(r, ncol, label);
            System.arraycopy(row, 0, matrix, i * ncol, ncol);
        }

        DMatrix result = new DMatrix(matrix, nrow, ncol);
        result.setLabel(labels);

        return result;
    }

    static Map<String, Object> generateParameters() {
        return new HashMap<String, Object>() {{
            put("eta", 1.0);
            put("max_depth", 6);
            put("silent", 1);
            put("objective", "binary:logistic");
        }};
    }

    static void generateTestModel() throws XGBoostError, IOException {
        byte nrow = 100;
        int ncol = NUM_DIMENSIONS;

        Map<String, Object> param = generateParameters();

        int round = 10;

        DMatrix trainMatrix = generateDMatrix(12345, nrow, ncol);
        DMatrix testMatrix = generateDMatrix(23456, nrow, ncol);

        Map<String, DMatrix> watchs = new LinkedHashMap<String, DMatrix>() {{
            put("train", trainMatrix);
            put("test", testMatrix);
        }};

        Booster booster = XGBoost.train(testMatrix, param, round, watchs, null, null);
        booster.saveModel(MODEL_PATH);

        trainMatrix.dispose();
        testMatrix.dispose();
        booster.dispose();
    }

    public static void main(String[] args) throws IOException, XGBoostError {
        generateTestModel();
    }
}
