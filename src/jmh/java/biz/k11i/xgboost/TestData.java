package biz.k11i.xgboost;

import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.util.Trainer;
import org.dmlc.xgboost4j.util.XGBoostError;

import java.io.IOException;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;
import java.util.stream.Collectors;

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

    static Iterable<Entry<String, Object>> generateParameters() {
        return new HashMap<String, Object>() {{
            put("eta", 1.0);
            put("max_depth", 6);
            put("silent", 1);
            put("objective", "binary:logistic");
        }}.entrySet();
    }

    static void generateTestModel() throws XGBoostError, IOException {
        byte nrow = 100;
        int ncol = NUM_DIMENSIONS;

        Iterable<Entry<String, Object>> param = generateParameters();

        byte round = 10;

        DMatrix trainMatrix = generateDMatrix(12345, nrow, ncol);
        DMatrix testMatrix = generateDMatrix(23456, nrow, ncol);

        List<Entry<String, DMatrix>> watchs = new LinkedHashMap<String, DMatrix>() {{
            put("train", trainMatrix);
            put("test", testMatrix);
        }}.entrySet().stream().collect(Collectors.toList());

        Booster booster = Trainer.train(param, testMatrix, round, watchs, null, null);
        booster.saveModel(MODEL_PATH);

        trainMatrix.delete();
        testMatrix.delete();
        booster.delete();
    }

    public static void main(String[] args) throws IOException, XGBoostError {
        generateTestModel();
    }
}
