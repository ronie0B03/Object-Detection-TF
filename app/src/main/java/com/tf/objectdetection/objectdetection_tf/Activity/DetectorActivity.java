package com.tf.objectdetection.objectdetection_tf.Activity;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Bundle;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.KeyEvent;
import android.view.View;
import android.widget.Toast;


import com.tf.objectdetection.objectdetection_tf.Interface.Classifier;
import com.tf.objectdetection.objectdetection_tf.Utils.ConnectionDetector;
import com.tf.objectdetection.objectdetection_tf.Models.GPSTracker;
import com.tf.objectdetection.objectdetection_tf.Models.Objext;
import com.tf.objectdetection.objectdetection_tf.Utils.OverlayView;
import com.tf.objectdetection.objectdetection_tf.Utils.OverlayView.DrawCallback;
import com.tf.objectdetection.objectdetection_tf.R;
import com.tf.objectdetection.objectdetection_tf.Tensorflow.TensorFlowMultiBoxDetector;
import com.tf.objectdetection.objectdetection_tf.Tensorflow.TensorFlowObjectDetectionAPIModel;
import com.tf.objectdetection.objectdetection_tf.Tensorflow.TensorFlowYoloDetector;
import com.tf.objectdetection.objectdetection_tf.env.BorderedText;
import com.tf.objectdetection.objectdetection_tf.env.ImageUtils;
import com.tf.objectdetection.objectdetection_tf.env.Logger;
import com.tf.objectdetection.objectdetection_tf.tracking.MultiBoxTracker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;
import java.util.Vector;

import io.nlopez.smartlocation.OnLocationUpdatedListener;
import io.nlopez.smartlocation.OnReverseGeocodingListener;
import io.nlopez.smartlocation.SmartLocation;


/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
    private static final String TAG = "DetectorActivity";
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged multibox model.
    private static final int MB_INPUT_SIZE = 224;
    private static final int MB_IMAGE_MEAN = 128;
    private static final float MB_IMAGE_STD = 128;
    private static final String MB_INPUT_NAME = "ResizeBilinear";
    private static final String MB_OUTPUT_LOCATIONS_NAME = "output_locations/Reshape";
    private static final String MB_OUTPUT_SCORES_NAME = "output_scores/Reshape";
    private static final String MB_MODEL_FILE = "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String MB_LOCATION_FILE = "file:///android_asset/multibox_location_priors.txt";

    private static final int TF_OD_API_INPUT_SIZE = 600;
    private static final String TF_OD_API_MODEL_FILE =
            "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";

    // Configuration values for tiny-yolo-voc. Note that the graph is not included with TensorFlow and
    // must be manually placed in the assets/ directory by the user.
    // Graphs and models downloaded from http://pjreddie.com/darknet/yolo/ may be converted e.g. via
    // DarkFlow (https://github.com/thtrieu/darkflow). Sample command:
    // ./flow --model cfg/tiny-yolo-voc.cfg --load bin/tiny-yolo-voc.weights --savepb --verbalise
    //private static final String YOLO_MODEL_FILE = "file:///android_asset/graph-tiny-yolo-voc.pb";
    private static final String YOLO_MODEL_FILE =  "file:///android_asset/ssd_mobilenet_v1_android_export.pb";
    private static final int YOLO_INPUT_SIZE = 416;
    private static final String YOLO_INPUT_NAME = "input";
    private static final String YOLO_OUTPUT_NAMES = "output";
    private static final int YOLO_BLOCK_SIZE = 32;


    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.  Optionally use legacy Multibox (trained using an older version of the API)
    // or YOLO.
    private enum DetectorMode {
        TF_OD_API, MULTIBOX, YOLO;
    }

    private static final DetectorMode MODE = DetectorMode.TF_OD_API;

    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.75f;
    private static final float MINIMUM_CONFIDENCE_MULTIBOX = 0.1f;
    private static final float MINIMUM_CONFIDENCE_YOLO = 0.25f;
    private static final boolean MAINTAIN_ASPECT = MODE == DetectorMode.YOLO;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(1280, 720);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Classifier detector;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private TextToSpeech textToSpeech;
    private boolean speak = true;
    HashMap<String, String> map = new HashMap<String, String>();
    private static final String UNIQUEID_TTS = "text2speechId";
    Geocoder geocoder;
    // GPSTracker class
    GPSTracker gps;
    double latitude, longitude;
    List<Address> addressList;
    ConnectionDetector connectionDetector;
    SmartLocation smartLocation;
    Location xlocation;


    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        Log.i(TAG, "onPreviewSizeChosen called");
        final float textSizePx =
                TypedValue.applyDimension(
                        TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        int cropSize = TF_OD_API_INPUT_SIZE;
        if (MODE == DetectorMode.YOLO) {
            detector =
                    TensorFlowYoloDetector.create(
                            getAssets(),
                            YOLO_MODEL_FILE,
                            YOLO_INPUT_SIZE,
                            YOLO_INPUT_NAME,
                            YOLO_OUTPUT_NAMES,
                            YOLO_BLOCK_SIZE);
            cropSize = YOLO_INPUT_SIZE;
        } else if (MODE == DetectorMode.MULTIBOX) {
            detector =
                    TensorFlowMultiBoxDetector.create(
                            getAssets(),
                            MB_MODEL_FILE,
                            MB_LOCATION_FILE,
                            MB_IMAGE_MEAN,
                            MB_IMAGE_STD,
                            MB_INPUT_NAME,
                            MB_OUTPUT_LOCATIONS_NAME,
                            MB_OUTPUT_SCORES_NAME);
            cropSize = MB_INPUT_SIZE;
        } else {
            try {
                detector = TensorFlowObjectDetectionAPIModel.create(
                        getAssets(), TF_OD_API_MODEL_FILE, TF_OD_API_LABELS_FILE, TF_OD_API_INPUT_SIZE);
                cropSize = TF_OD_API_INPUT_SIZE;

                Log.i(TAG, "created a detector");
            } catch (final IOException e) {
                Log.e(TAG, e.getMessage());
                LOGGER.e("Exception initializing classifier!", e);
                Toast toast =
                        Toast.makeText(
                                getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
                toast.show();
                finish();
            }
        }

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        Log.i(TAG, "width:" + previewWidth + ",height:" + previewHeight);

//        sensorOrientation = rotation + getScreenOrientation();
        sensorOrientation = 0;

        Log.i(TAG, "rotation:" + rotation + ",screenorientation:" + getScreenOrientation());
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        cropSize, cropSize,
                        sensorOrientation, MAINTAIN_ASPECT);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        tracker.draw(canvas);
                        if (isDebug()) {
                            tracker.drawDebug(canvas);
                        }
                    }
                });

        addCallback(
                new DrawCallback() {
                    @Override
                    public void drawCallback(final Canvas canvas) {
                        if (!isDebug()) {
                            return;
                        }
                        final Bitmap copy = cropCopyBitmap;
                        if (copy == null) {
                            return;
                        }

                        final int backgroundColor = Color.argb(100, 0, 0, 0);
                        canvas.drawColor(backgroundColor);

                        final Matrix matrix = new Matrix();
                        final float scaleFactor = 2;
                        matrix.postScale(scaleFactor, scaleFactor);
                        matrix.postTranslate(
                                canvas.getWidth() - copy.getWidth() * scaleFactor,
                                canvas.getHeight() - copy.getHeight() * scaleFactor);
                        canvas.drawBitmap(copy, matrix, new Paint());

                        final Vector<String> lines = new Vector<String>();
                        if (detector != null) {
                            final String statString = detector.getStatString();
                            final String[] statLines = statString.split("\n");
                            for (final String line : statLines) {
                                lines.add(line);
                            }
                        }
                        lines.add("");

                        lines.add("Frame: " + previewWidth + "x" + previewHeight);
                        lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                        lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                        lines.add("Rotation: " + sensorOrientation);
                        lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                        borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                    }
                });
    }

    OverlayView trackingOverlay;

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                new Runnable() {
                    @Override
                    public void run() {
                        LOGGER.i("Running detection on image " + currTimestamp);
                        final long startTime = SystemClock.uptimeMillis();
                        final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);

                        lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

                        cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                        final Canvas canvas = new Canvas(cropCopyBitmap);
                        final Paint paint = new Paint();
                        paint.setColor(Color.RED);
                        paint.setStyle(Style.STROKE);
                        paint.setStrokeWidth(2.0f);

                        float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                        switch (MODE) {
                            case TF_OD_API:
                                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                                break;
                            case MULTIBOX:
                                minimumConfidence = MINIMUM_CONFIDENCE_MULTIBOX;
                                break;
                            case YOLO:
                                minimumConfidence = MINIMUM_CONFIDENCE_YOLO;
                                break;
                        }

                        final List<Classifier.Recognition> mappedRecognitions =
                                new LinkedList<Classifier.Recognition>();

                        for (final Classifier.Recognition result : results) {
                            final RectF location = result.getLocation();
                            if (location != null && result.getConfidence() >= minimumConfidence) {
                                canvas.drawRect(location, paint);
                                cropToFrameTransform.mapRect(location);
                                result.setLocation(location);
                                mappedRecognitions.add(result);

                            }

                        }
                        TTSpeech(mappedRecognitions);

                        tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                        trackingOverlay.postInvalidate();

                        requestRender();
                        computingDetection = false;
                    }
                });
    }

    String xAddress = "";


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        geocoder = new Geocoder(this, Locale.getDefault());
        connectionDetector = new ConnectionDetector(this);

        SmartLocation.with(this).location().start(new OnLocationUpdatedListener() {
            @Override
            public void onLocationUpdated(Location location) {
                //pass
                xlocation = location;
                SmartLocation.with(DetectorActivity.this).geocoding()
                        .reverse(location, new OnReverseGeocodingListener() {
                            @Override
                            public void onAddressResolved(Location location, List<Address> list) {
                                try {
                                    String feature = list.get(0).getFeatureName();
                                    String locality = list.get(0).getLocality();
                                    String subAdmin = list.get(0).getSubAdminArea();

                                    String address = feature + ", " + locality + ", " + subAdmin;

                                    String fullAddress = "Your current location is, " + address;

                                    Log.i(TAG, fullAddress);
                                    xAddress = fullAddress;

                                } catch (IndexOutOfBoundsException e) {
                                    Log.e(TAG, e.getMessage());

                                }


                            }
                        });
            }
        });


        textToSpeech = new TextToSpeech(getApplicationContext(), new TextToSpeech.OnInitListener() {
            @Override
            public void onInit(int status) {

                if (status == TextToSpeech.SUCCESS) {
                    int ttsLang = textToSpeech.setLanguage(Locale.US);
                    if (ttsLang == TextToSpeech.LANG_MISSING_DATA
                            || ttsLang == TextToSpeech.LANG_NOT_SUPPORTED) {
                        Log.e(TAG, "The Language is not supported!");
                    } else {
                        Log.i(TAG, "Language Supported.");
                    }
                    Log.i(TAG, "Initialization success.");
                } else {
                    Toast.makeText(getApplicationContext(), "TTS Initialization failed!", Toast.LENGTH_SHORT).show();
                }
            }
        });

        textToSpeech.setOnUtteranceProgressListener(new UtteranceProgressListener() {
            @Override
            public void onStart(String s) {
                speak = false;

            }

            @Override
            public void onDone(String s) {
                speak = true;

            }

            @Override
            public void onError(String s) {
                speak = true;

            }
        });


    }


    private void getLocation() {
        if (!connectionDetector.isConnectingToInternet()) {
            String error = "Please connect to the internet to use the Locator";
            Toast.makeText(this, error, Toast.LENGTH_SHORT).show();

            if (!error.equals("")) {
                map.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, UNIQUEID_TTS);
                textToSpeech.speak(error, TextToSpeech.QUEUE_FLUSH, map);

            } else {
                speak = true;
            }
            return;
        }


        SmartLocation.with(this).location()
                .oneFix()
                .start(new OnLocationUpdatedListener() {
                    @Override
                    public void onLocationUpdated(Location location) {
//                        Toast.makeText(DetectorActivity.this, location.toString(), Toast.LENGTH_SHORT).show();
                        xlocation = location;
                    }
                });


        if (xlocation != null) {
            SmartLocation.with(this).geocoding()
                    .reverse(xlocation, new OnReverseGeocodingListener() {
                        @Override
                        public void onAddressResolved(Location location, List<Address> list) {
                            try {
                                String feature = list.get(0).getFeatureName();
                                String locality = list.get(0).getLocality();
                                String subAdmin = list.get(0).getSubAdminArea();

                                String address = feature + ", " + locality + ", " + subAdmin;

                                String fullAddress = "Your current location is, " + address;

                                Log.i(TAG, fullAddress);
                                Log.i(TAG, list.get(0).toString());
                                Toast.makeText(DetectorActivity.this, address, Toast.LENGTH_SHORT).show();

                                if (!fullAddress.equals("")) {
                                    map.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, UNIQUEID_TTS);
                                    textToSpeech.speak(fullAddress, TextToSpeech.QUEUE_FLUSH, map);

                                } else {
                                    speak = true;
                                }
                            } catch (IndexOutOfBoundsException e) {
                                if (!xAddress.equals("")) {
                                    map.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, UNIQUEID_TTS);
                                    textToSpeech.speak(xAddress, TextToSpeech.QUEUE_FLUSH, map);

                                } else {
                                    speak = true;
                                }
                                Log.e(TAG, e.getMessage());

                            }


                        }
                    });
        }


    }


    public void onClickLocation(View view) {
        getLocation();
    }

    @Override
    public synchronized void onPause() {
        super.onPause();

    }

    @Override
    public synchronized void onStop() {
        super.onStop();
        SmartLocation.with(DetectorActivity.this).location().stop();
        SmartLocation.with(DetectorActivity.this).geocoding().stop();
        Log.d(TAG, "Stopping the location service");
    }

    @Override
    public boolean onKeyDown(final int keyCode, final KeyEvent event) {
        if (keyCode == KeyEvent.KEYCODE_VOLUME_DOWN || keyCode == KeyEvent.KEYCODE_VOLUME_UP
                || keyCode == KeyEvent.KEYCODE_BUTTON_L1 || keyCode == KeyEvent.KEYCODE_DPAD_CENTER) {
            getLocation();
            return true;
        }
        return super.onKeyDown(keyCode, event);
    }


    private void TTSpeech(List<Classifier.Recognition> objects) {
        if (speak) {
            speak = false;
            String text = "";
            List<Classifier.Recognition> left = new ArrayList<>();
            List<Classifier.Recognition> centre = new ArrayList<>();
            List<Classifier.Recognition> right = new ArrayList<>();

            //SORT THE OBJECTS
            for (Classifier.Recognition object : objects) {
                int location = (int) object.getLocation().centerX();
                int width = previewWidth / 3;
                if (0 < location && width > location) {
                    left.add(object);
                } else if (width < location && (width * 2) > location) {
                    centre.add(object);
                } else if ((width * 2) < location && (width * 3) > location) {
                    right.add(object);
                }
            }

            //GENERATE THE TEXT
            if (left.size() != 0) {
                text += "in your left,";
                text += generateObjectTitle(left);
            }

            if (centre.size() != 0) {
                text += ", in your center,";
                text += generateObjectTitle(centre);

            }
            if (right.size() != 0) {
                text += ", in your right,";
                text += generateObjectTitle(right);

            }

            if (!text.equals("")) {
                map.put(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, UNIQUEID_TTS);
                textToSpeech.speak(text, TextToSpeech.QUEUE_FLUSH, map);

            } else {
                speak = true;
            }

        }


    }


    private String generateObjectTitle(List<Classifier.Recognition> objects) {
        List<Objext> objexts = new ArrayList<>();
        for (Classifier.Recognition hold : objects) {
            if (objexts.size() == 0) {
                objexts.add(new Objext(hold.getTitle(), 1));
            } else {
                boolean ok = true;
                for (Objext xhold : objexts) {
                    if (xhold.getTitle().equals(hold.getTitle())) {
                        xhold.increment();
                        ok = false;
                        break;
                    }
                }
                if (ok) {
                    objexts.add(new Objext(hold.getTitle(), 1));
                }


            }
        }
        String text = "";
        for (Objext hold : objexts) {
            text += hold.toString();
        }

        return text;
    }


    @Override
    protected int getLayoutId() {
        return R.layout.fragment_camera;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    public void onSetDebug(final boolean debug) {
        detector.enableStatLogging(debug);
    }


}
