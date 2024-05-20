package com.example.openwakeword;
import android.content.res.AssetManager;
import android.util.Log;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Random;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;


class ONNXModelRunner {

    private static final int BATCH_SIZE = 1; // Replace with your batch size

    AssetManager assetManager;
    OrtSession hey_nugget_session;
    OrtEnvironment hey_nugget_env = OrtEnvironment.getEnvironment();
    public ONNXModelRunner(AssetManager assetManager) throws IOException, OrtException {
        this.assetManager=assetManager;

        try {
            hey_nugget_session = hey_nugget_env.createSession(readModelFile(assetManager, "hey_nugget_new.onnx"));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // Load the ONNX model from the assets folder

    }

    public float[][] get_mel_spectrogram(float[] inputArray) throws OrtException, IOException {
        OrtSession session;
        try (InputStream modelInputStream = assetManager.open("melspectrogram.onnx")) {
            byte[] modelBytes = new byte[modelInputStream.available()];
            modelInputStream.read(modelBytes);
            session = OrtEnvironment.getEnvironment().createSession(modelBytes);
        }
        float[][] outputArray=null;
        int SAMPLES=inputArray.length;
        // Convert the input array to ONNX Tensor
        FloatBuffer floatBuffer = FloatBuffer.wrap(inputArray);
        OnnxTensor inputTensor = OnnxTensor.createTensor(OrtEnvironment.getEnvironment(), floatBuffer, new long[]{BATCH_SIZE, SAMPLES});

        // Run the model
        // Adjust this based on the actual expected output shape
        try (OrtSession.Result results = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), inputTensor))) {

            float[][][][] outputTensor = (float[][][][]) results.get(0).getValue();
            // Here you need to cast the output appropriately
//            Object outputObject = outputTensor.getValue();

            // Check the actual type of 'outputObject' and cast accordingly
            // The following is an assumed cast based on your error message

            float[][] squeezed=squeeze(outputTensor);
            outputArray=applyMelSpecTransform(squeezed);


        }
        catch (Exception e)
        {
            e.printStackTrace();

        }
        finally {
            if (inputTensor != null) inputTensor.close();
            if (session!=null) session.close();
        }
        OrtEnvironment.getEnvironment().close();
        return outputArray;
    }
    public static float[][] squeeze(float[][][][] originalArray) {
        float[][] squeezedArray = new float[originalArray[0][0].length][originalArray[0][0][0].length];
        for (int i = 0; i < originalArray[0][0].length; i++) {
            for (int j = 0; j < originalArray[0][0][0].length; j++) {
                squeezedArray[i][j] = originalArray[0][0][i][j];
            }
        }

        return squeezedArray;
    }
    public static float[][] applyMelSpecTransform(float[][] array) {
        float[][] transformedArray = new float[array.length][array[0].length];

        for (int i = 0; i < array.length; i++) {
            for (int j = 0; j < array[i].length; j++) {
                transformedArray[i][j] = array[i][j] / 10.0f + 2.0f;
            }
        }

        return transformedArray;
    }

    public float[][] generateEmbeddings(float[][][][] input) throws OrtException, IOException {
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        InputStream is = assetManager.open("embedding_model.onnx");
        byte[] model = new byte[is.available()];
        is.read(model);
        is.close();

        OrtSession sess = env.createSession(model);
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, input);
        try (OrtSession.Result results = sess.run(Collections.singletonMap("input_1", inputTensor))) {
            // Extract the output tensor
            float[][][][] rawOutput = (float[][][][]) results.get(0).getValue();

            // Assuming the output shape is (41, 1, 1, 96), and we want to reshape it to (41, 96)
            float[][] reshapedOutput = new float[rawOutput.length][rawOutput[0][0][0].length];
            for (int i = 0; i < rawOutput.length; i++) {
                System.arraycopy(rawOutput[i][0][0], 0, reshapedOutput[i], 0, rawOutput[i][0][0].length);
            }
            return reshapedOutput;
        } catch (Exception e) {
            Log.d("exception", "not_predicted " + e.getMessage());
        }
        finally {
            if (inputTensor != null) inputTensor.close(); // You're doing this, which is good.
            if (sess != null) sess.close(); // This should be added to ensure the session is also closed.
        }
        env.close();
        return null;
    }

    public String predictWakeWord(float[][][] inputArray) throws OrtException {
        float[][] result = new float[0][];
        String resultant="";


        OnnxTensor inputTensor = null;

        try {
            // Create a tensor from the input array
            inputTensor = OnnxTensor.createTensor(hey_nugget_env, inputArray);
            // Run the inference
            OrtSession.Result outputs = hey_nugget_session.run(Collections.singletonMap(hey_nugget_session.getInputNames().iterator().next(), inputTensor));
            // Extract the output tensor, convert it to the desired type
             result=(float[][]) outputs.get(0).getValue();
            resultant= String.format("%.5f", (double) result[0][0]);

        } catch (OrtException e) {
            e.printStackTrace();
        }
        finally {
            if (inputTensor != null) inputTensor.close();
             // Add this to ensure the session is properly closed.
        }
        return resultant;
    }
    private byte[] readModelFile(AssetManager assetManager, String filename) throws IOException {
        try (InputStream is = assetManager.open(filename)) {
            byte[] buffer = new byte[is.available()];
            is.read(buffer);
            return buffer;
        }
    }

}


public class Model {
    int n_prepared_samples=1280;
    int sampleRate=16000;
    int melspectrogramMaxLen= 10*97;
    int feature_buffer_max_len=120;
    ONNXModelRunner modelRunner;
    float[][] featureBuffer;
    ArrayDeque<Float> raw_data_buffer=new ArrayDeque<>(sampleRate * 10);;
    float[] raw_data_remainder = new float[0];
    float[][] melspectrogramBuffer;
    int accumulated_samples=0;
    Model(ONNXModelRunner modelRunner) {
        melspectrogramBuffer = new float[76][32];
        for (int i = 0; i < melspectrogramBuffer.length; i++) {
            for (int j = 0; j < melspectrogramBuffer[i].length; j++) {
                melspectrogramBuffer[i][j] = 1.0f; // Assign 1.0f to simulate numpy.ones
            }
        }
        this.modelRunner=modelRunner;
        try{

            this.featureBuffer = this._getEmbeddings(this.generateRandomIntArray(16000 * 4), 76, 8);

        }
    catch (Exception e)
    {

        System.out.print(e.getMessage());
    }

    }

    public float[][][] getFeatures(int nFeatureFrames, int startNdx) {
        int endNdx;
        if (startNdx != -1) {
            endNdx = (startNdx + nFeatureFrames != 0) ? (startNdx + nFeatureFrames) : featureBuffer.length;
        } else {
            startNdx = Math.max(0, featureBuffer.length - nFeatureFrames); // Ensure startNdx is not negative
            endNdx = featureBuffer.length;
        }

        int length = endNdx - startNdx;
        float[][][] result = new float[1][length][featureBuffer[0].length]; // Assuming the second dimension has fixed size.

        for (int i = 0; i < length; i++) {
            System.arraycopy(featureBuffer[startNdx + i], 0, result[0][i], 0, featureBuffer[startNdx + i].length);
        }

        return result;
    }

    // Java equivalent to _get_embeddings method
    private float[][] _getEmbeddings(float[] x, int windowSize, int stepSize) throws OrtException, IOException {

        float[][] spec = this.modelRunner.get_mel_spectrogram(x); // Assuming this method exists and returns float[][]
        ArrayList<float[][]> windows = new ArrayList<>();

        for (int i = 0; i <= spec.length - windowSize; i += stepSize) {
            float[][] window = new float[windowSize][spec[0].length];

            for (int j = 0; j < windowSize; j++) {
                System.arraycopy(spec[i + j], 0, window[j], 0, spec[0].length);
            }

            // Check if the window is full-sized (not truncated)
            if (window.length == windowSize) {
                windows.add(window);
            }
        }

        // Convert ArrayList to array and add the required extra dimension
        float[][][][] batch = new float[windows.size()][windowSize][spec[0].length][1];
        for (int i = 0; i < windows.size(); i++) {
            for (int j = 0; j < windowSize; j++) {
                for (int k = 0; k < spec[0].length; k++) {
                    batch[i][j][k][0] = windows.get(i)[j][k];  // Add the extra dimension here
                }
            }
        }

        try {
           float[][]  result= modelRunner.generateEmbeddings(batch);
           return result;
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        // Assuming embeddingModelPredict is defined and returns float[][]
    }

    // Utility function to generate random int array, equivalent to np.random.randint
    private float[] generateRandomIntArray(int size) {
        float[] arr = new float[size];
        Random random = new Random();
        for (int i = 0; i < size; i++) {
            arr[i] = (float) random.nextInt(2000) - 1000; // range [-1000, 1000)
        }
        return arr;
    }
    public void bufferRawData(float[] x) { // Change double[] to match your actual data type
        // Check if input x is not null
        if (x != null) {
            // Check if raw_data_buffer has enough space, if not, remove old data
            while (raw_data_buffer.size() + x.length > sampleRate * 10) {
                raw_data_buffer.poll(); // or pollFirst() - removes and returns the first element of this deque
            }
            for (float value : x) {
                raw_data_buffer.offer(value); // or offerLast() - Inserts the specified element at the end of this deque
            }
        }
    }

    public void streamingMelSpectrogram(int n_samples) {
        if (raw_data_buffer.size() < 400) {
            throw new IllegalArgumentException("The number of input frames must be at least 400 samples @ 16kHz (25 ms)!");
        }

        // Converting the last n_samples + 480 (3 * 160) samples from raw_data_buffer to an ArrayList
        float[] tempArray = new float[n_samples + 480]; // 160 * 3 = 480
        Object[] rawDataArray = raw_data_buffer.toArray();
        for (int i = Math.max(0, rawDataArray.length - n_samples - 480); i < rawDataArray.length; i++) {
            tempArray[i - Math.max(0, rawDataArray.length - n_samples - 480)] = (Float) rawDataArray[i];
        }

        // Assuming getMelSpectrogram returns a two-dimensional float array
        float[][] new_mel_spectrogram ;
        try {
            new_mel_spectrogram = modelRunner.get_mel_spectrogram(tempArray);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        // Combine existing melspectrogram_buffer with new_mel_spectrogram
        float[][] combined = new float[this.melspectrogramBuffer.length + new_mel_spectrogram.length][];

        System.arraycopy(this.melspectrogramBuffer, 0, combined, 0, this.melspectrogramBuffer.length);
        System.arraycopy(new_mel_spectrogram, 0, combined, this.melspectrogramBuffer.length, new_mel_spectrogram.length);
        this.melspectrogramBuffer = combined;

        // Trim the melspectrogram_buffer if it exceeds the max length
        if (this.melspectrogramBuffer.length > melspectrogramMaxLen) {
            float[][] trimmed = new float[melspectrogramMaxLen][];
            System.arraycopy(this.melspectrogramBuffer, this.melspectrogramBuffer.length - melspectrogramMaxLen, trimmed, 0, melspectrogramMaxLen);
            this.melspectrogramBuffer = trimmed;
        }

    }

    public int streaming_features(float[] audiobuffer) {
        int processed_samples = 0;
        this.accumulated_samples=0;
        if (raw_data_remainder.length != 0) {
            // Create a new array to hold the result of concatenation
            float[] concatenatedArray = new float[raw_data_remainder.length + audiobuffer.length];

            // Copy elements from raw_data_remainder to the new array
            System.arraycopy(raw_data_remainder, 0, concatenatedArray, 0, raw_data_remainder.length);

            // Copy elements from x to the new array, starting right after the last element of raw_data_remainder
            System.arraycopy(audiobuffer, 0, concatenatedArray, raw_data_remainder.length, audiobuffer.length);

            // Assign the concatenated array back to x
            audiobuffer = concatenatedArray;

            // Reset raw_data_remainder to an empty array
            raw_data_remainder = new float[0];
        }

        if (this.accumulated_samples + audiobuffer.length >= 1280) {
            int remainder = (this.accumulated_samples + audiobuffer.length) % 1280;
            if (remainder != 0) {
                // Create an array for x_even_chunks that excludes the last 'remainder' elements of 'x'
                float[] x_even_chunks = new float[audiobuffer.length - remainder];
                System.arraycopy(audiobuffer, 0, x_even_chunks, 0, audiobuffer.length - remainder);

                // Buffer the even chunks of data
                this.bufferRawData(x_even_chunks);

                // Update accumulated_samples by the length of x_even_chunks
                this.accumulated_samples += x_even_chunks.length;

                // Set raw_data_remainder to the last 'remainder' elements of 'x'
                this.raw_data_remainder = new float[remainder];
                System.arraycopy(audiobuffer, audiobuffer.length - remainder, this.raw_data_remainder, 0, remainder);
            } else if (remainder == 0) {
                // Buffer the entire array 'x'
                this.bufferRawData(audiobuffer);

                // Update accumulated_samples by the length of 'x'
                this.accumulated_samples += audiobuffer.length;

                // Set raw_data_remainder to an empty array
                this.raw_data_remainder = new float[0];
            }
        } else {
            this.accumulated_samples += audiobuffer.length;
            this.bufferRawData(audiobuffer); // Adapt this method according to your class
        }


        if (this.accumulated_samples >= 1280 && this.accumulated_samples % 1280 == 0) {

            this.streamingMelSpectrogram(this.accumulated_samples);

            float[][][][] x = new float[1][76][32][1];

            for (int i = (accumulated_samples / 1280) - 1; i >= 0; i--) {

                int ndx = -8 * i;
                if (ndx == 0) {
                    ndx = melspectrogramBuffer.length;
                }
                // Calculate start and end indices for slicing
                int start = Math.max(0, ndx - 76);
                int end = ndx;

                for (int j = start, k = 0; j < end; j++, k++) {
                    for (int w = 0; w < 32; w++) {
                        x[0][k][w][0] = (float) melspectrogramBuffer[j][w];
                    }
                }
                if (x[0].length== 76)
                {
                    try {
                        float[][] newFeatures=modelRunner.generateEmbeddings(x);
                        if (featureBuffer == null) {
                            featureBuffer = newFeatures;
                        } else {
                            int totalRows = featureBuffer.length + newFeatures.length;
                            int numColumns = featureBuffer[0].length; // Assuming all rows have the same length
                            float[][] updatedBuffer = new float[totalRows][numColumns];

                            // Copy original featureBuffer into updatedBuffer
                            for (int l = 0; l< featureBuffer.length; l++) {
                                System.arraycopy(featureBuffer[l], 0, updatedBuffer[l], 0, featureBuffer[l].length);
                            }

                            // Copy newFeatures into the updatedBuffer, starting after the last original row
                            for (int k = 0; k < newFeatures.length; k++) {
                                System.arraycopy(newFeatures[k], 0, updatedBuffer[k + featureBuffer.length], 0, newFeatures[k].length);
                            }

                            featureBuffer = updatedBuffer;
                        }

                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }
            }
            processed_samples=this.accumulated_samples;
            this.accumulated_samples=0;

        }
        if (featureBuffer.length > feature_buffer_max_len) {
            float[][] trimmedFeatureBuffer = new float[feature_buffer_max_len][featureBuffer[0].length];

            // Copy the last featureBufferMaxLen rows of featureBuffer into trimmedFeatureBuffer
            for (int i = 0; i < feature_buffer_max_len; i++) {
                trimmedFeatureBuffer[i] = featureBuffer[featureBuffer.length - feature_buffer_max_len + i];
            }

            // Update featureBuffer to point to the new trimmedFeatureBuffer
            featureBuffer = trimmedFeatureBuffer;
        }
        return processed_samples != 0 ? processed_samples : this.accumulated_samples;



    }

    public String predict_WakeWord(float[] audiobuffer){

        n_prepared_samples=this.streaming_features(audiobuffer);
        float[][][] res=this.getFeatures(16,-1);
        String result="";
        try {
            result=modelRunner.predictWakeWord(res);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
        return  result;
        }
    }

