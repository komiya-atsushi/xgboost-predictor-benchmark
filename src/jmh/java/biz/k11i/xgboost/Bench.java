package biz.k11i.xgboost;

import biz.k11i.xgboost.util.FVec;
import org.dmlc.xgboost4j.Booster;
import org.dmlc.xgboost4j.DMatrix;
import org.dmlc.xgboost4j.util.XGBoostError;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.infra.Blackhole;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Random;

public class Bench {
    @State(Scope.Benchmark)
    public static class PredictionState {
        static final int NUM_ROWS = 100;

        final Booster booster;
        final Predictor predictor;
        final float[][] data;
        final float[] matrix;

        public PredictionState() {
            try (FileInputStream fis = new FileInputStream(TestData.MODEL_PATH)) {
                    this.booster = new Booster(TestData.generateParameters(), TestData.MODEL_PATH);
                    this.predictor = new Predictor(fis);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

            this.data = new float[NUM_ROWS][];
            this.matrix = new float[NUM_ROWS * TestData.NUM_DIMENSIONS];

            Random r = new Random(34567);

            for(int i = 0; i < NUM_ROWS; ++i) {
                this.data[i] = TestData.generateRow(r, TestData.NUM_DIMENSIONS, i % 2);
                System.arraycopy(this.data[i], 0, this.matrix, i * TestData.NUM_DIMENSIONS, TestData.NUM_DIMENSIONS);
            }
        }
    }

    public static class Load {
        @Benchmark
        public void xgboost4j(Blackhole bh) throws XGBoostError {
            bh.consume(new Booster(TestData.generateParameters(), TestData.MODEL_PATH));
        }

        @Benchmark
        public void xgboostPredictor(Blackhole bh) throws IOException {
            try (FileInputStream fis = new FileInputStream(TestData.MODEL_PATH);
                 BufferedInputStream bis = new BufferedInputStream(fis)) {
                bh.consume(new Predictor(bis));
            }
        }
    }

    public static class SinglePrediction {
        @Benchmark
        public void xgboost4j(PredictionState model) throws XGBoostError {
            DMatrix fv = new DMatrix(model.data[0], 1, TestData.NUM_DIMENSIONS);
            model.booster.predict(fv);
            fv.delete();
        }

        @Benchmark
        public void xgboostPredictor(PredictionState model) {
            FVec fv = FVec.Transformer.fromArray(model.data[0], false);
            model.predictor.predict(fv);
        }
    }

    public static class BatchPrediction {
        @Benchmark
        public void xgboost4j(PredictionState model) throws XGBoostError {
            DMatrix fv = new DMatrix(model.matrix, PredictionState.NUM_ROWS, TestData.NUM_DIMENSIONS);
            model.booster.predict(fv);
            fv.delete();
        }

        @Benchmark
        public void xgboostPredictor(PredictionState model) {
            for (float[] values : model.data) {
                FVec fv = FVec.Transformer.fromArray(values, false);
                model.predictor.predict(fv);
            }
        }
    }

    public static class LeafPrediction {
        @Benchmark
        public void xgboost4j(PredictionState model) throws XGBoostError {
            DMatrix fv = new DMatrix(model.data[0], 1, TestData.NUM_DIMENSIONS);
            model.booster.predict(fv, 0, true);
            fv.delete();
        }

        @Benchmark
        public void xgboostPredictor(PredictionState model) {
            FVec fv = FVec.Transformer.fromArray(model.data[0], false);
            model.predictor.predictLeaf(fv);
        }
    }
}
