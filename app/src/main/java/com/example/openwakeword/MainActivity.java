package com.example.openwakeword;

import android.Manifest;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Process;
import android.util.Log;
import android.widget.TextView;

import androidx.activity.EdgeToEdge;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;




public class MainActivity extends AppCompatActivity {
    private static final int PERMISSION_REQUEST_RECORD_AUDIO = 200;
    TextView melspec,predicted_text;
    ONNXModelRunner modelRunner;
    AssetManager assetManager;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        melspec=findViewById(R.id.melspec);
        predicted_text=findViewById(R.id.predicted);
        assetManager = this.getAssets();
        try {
            modelRunner = new ONNXModelRunner(assetManager);
        }
        catch (Exception e) {
            Log.d("exception",e.getMessage());
        }
        if (checkAndRequestPermissions()) {

            Model model=new Model(modelRunner);
            AudioRecorderThread recorder = new AudioRecorderThread(this,melspec,predicted_text,modelRunner,model);
            recorder.start(); // Start recording
            // recorder.stopRecording();
        }
    }
    private boolean checkAndRequestPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, PERMISSION_REQUEST_RECORD_AUDIO);
            return false;
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case PERMISSION_REQUEST_RECORD_AUDIO: {
                if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    // Permission was granted, you can continue your operation
                } else {
                    // Permission denied, you can disable the functionality that depends on this permission.
                }
                break;
            }
        }
    }
}

class AudioRecorderThread extends Thread {
    private static final int SAMPLE_RATE = 16000;
    private static final int CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO;
    private static final int AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT;
    private AudioRecord audioRecord;
    private boolean isRecording = false;
    private Context context;
    private TextView melspectext,predicted_text;
    ONNXModelRunner modelRunner;
    Model model;
    AudioRecorderThread (Context context,TextView melspec,TextView predicted_text,ONNXModelRunner modelRunner, Model model)
    {
        this.context=context;
        this.melspectext=melspec;
        this.modelRunner=modelRunner;
        this.model=model;
        this.predicted_text=predicted_text;

    }
    @SuppressLint("MissingPermission")
    @Override
    public void run() {
        Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);

        // Ensure the buffer size is at least as large as the chunk size needed
        int minBufferSize = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT);
        int bufferSizeInShorts = 1280; // This is your 'chunk size' in terms of shorts
        if (minBufferSize / 2 < bufferSizeInShorts) {
            minBufferSize = bufferSizeInShorts * 2; // Ensure buffer is large enough, adjusting if necessary
        }

        audioRecord = new AudioRecord(MediaRecorder.AudioSource.MIC, SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT, minBufferSize);

        if (audioRecord.getState() != AudioRecord.STATE_INITIALIZED) {
            // Initialization error handling
            return;
        }

        short[] audioBuffer = new short[bufferSizeInShorts]; // Allocate buffer for 'chunk size' shorts
        audioRecord.startRecording();
        isRecording = true;

        while (isRecording) {
            // Reading data from the microphone in chunks
            audioRecord.read(audioBuffer, 0, audioBuffer.length);
            float[] floatBuffer = new float[audioBuffer.length];

            // Convert each short to float
            for (int i = 0; i < audioBuffer.length; i++) {
                // Convert by dividing by the maximum value of short to normalize
                floatBuffer[i] = audioBuffer[i] / 32768.0f; // Normalize to range -1.0 to 1.0 if needed
            }
            String res=model.predict_WakeWord(floatBuffer);
            ((Activity) context).runOnUiThread(new Runnable() {
                public void run() {
                    melspectext.setText(res);
                    if (Double.parseDouble(res) > 0.05) {
                        predicted_text.setText("Wake Word Detected!");

                    }
                    else{
                        predicted_text.setText("");
                    }
                }
            });
        }

        releaseResources();
    }

    public void stopRecording() {
        isRecording = false;
    }

    private void releaseResources() {
        if (audioRecord != null) {
            audioRecord.stop();
            audioRecord.release();
            audioRecord = null;
        }
    }
}