/*
Example of tensorflow-lite in Android
This example uses model trained for classifying skin mole image for melanoma and non-melanoma
NasNetMobile model is used.
 */
package ccj.com.tensorflow_liteinandroidexample;

import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.os.Handler;
import android.os.Message;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.theartofdev.edmodo.cropper.CropImage;
import com.theartofdev.edmodo.cropper.CropImageView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    Bitmap Image;
    public static Interpreter tflite;
    private static String MODEL_PATH;//Used to store model path
    float output[][]=new float[1][1];//For storing result

    Handler handler1 = new Handler()
    {
        @Override
        public void handleMessage(Message msg)
        {
            switch (msg.what)
            {
                case 0:
                    //Model is loaded make choose image button visible
                    Button button = findViewById(R.id.button);
                    button.setVisibility(View.VISIBLE);
                    break;
                case 1:
                    //Computation is over display the output
                    TextView tv = findViewById(R.id.textView);
                    tv.setText("Melanoma Probability:"+output[0][0]);
            }


        }
    };

    //Function to load model
    private void load_model()
    {

        //Thread for loading model
        Runnable load_model = new Runnable()
        {

            @Override
            public void run()
            {
                try
                {
                    tflite = new Interpreter(loadModelFile(getAssets(), MODEL_PATH));
                }
                catch (Exception e) {}
                handler1.sendEmptyMessage(0);
            }
        };

        Thread model_loading_thread=new Thread(load_model);
        model_loading_thread.start();
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Button button = findViewById(R.id.button);
        button.setVisibility(View.GONE);
        tflite=null;
        MODEL_PATH = "my_model.tflite";
        Image=null;

        //Load model as soon as activity is created.
        load_model();
    }

    public void onClick(View view) {
        //Activity for choosing image
        CropImage.activity()
                .setGuidelines(CropImageView.Guidelines.ON)
                .setCropShape(CropImageView.CropShape.OVAL)
                .setAspectRatio(1, 1)
                .setRequestedSize(224,224)
                .setFixAspectRatio(true)
                .start(this);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        // handle result of CropImageActivity
        if (requestCode == CropImage.CROP_IMAGE_ACTIVITY_REQUEST_CODE)
        {
            CropImage.ActivityResult result = CropImage.getActivityResult(data);
            if (resultCode == RESULT_OK)
            {
                try
                {
                    Image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), result.getUri());
                }
                catch (IOException e) {}
                ImageView imageView= findViewById(R.id.imageView);
                imageView.setImageBitmap(Image);
                //Running Thread for doing prediction as it is heavy operation
                Runnable predict_r = new Runnable() {
                    @Override
                    public void run() {
                        ByteBuffer byteBuffer = convertBitmapToByteBuffer(Image);
                        MainActivity.tflite.run(byteBuffer,output);
                        handler1.sendEmptyMessage(1);
                    }
                };
                Thread predict = new Thread(predict_r);
                predict.start();

            }
        }
    }

    //Returns the required path for loading model.
    private MappedByteBuffer loadModelFile(AssetManager assetManager, String modelPath) throws IOException
    {
        AssetFileDescriptor fileDescriptor = assetManager.openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }


    //Convert bitmap to bytebuffer
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap)
    {

        //Initialize space for byte buffer FLOAT_SIZE*BATCH_SIZE*INPUT_SHAPE*INPUT_SHAPE*NUMBER_OF_CHANNELS
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4*1*224*224*3);

        byteBuffer.order(ByteOrder.nativeOrder());

        //Array for holding pixels values. Pixel values are stored in packed integer.
        int[] intValues = new int[224*224];

        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        int pixel = 0;
        float divi = 127.5f;

        for (int i = 0; i < 224; ++i)
        {
            for (int j = 0; j < 224; ++j)
            {
               /*Preprocessing for nasNet
               pixel = pixel/127.5
               pixel = pixel-1
                */
                final int val = intValues[pixel++];

                //Decoding pixel value,preprocessing and putting into bytebuffer.

                byteBuffer.putFloat((((val >> 16) & 0xFF))/divi-1);
                byteBuffer.putFloat((((val >> 8) & 0xFF))/divi-1);
                byteBuffer.putFloat((((val) & 0xFF))/divi-1);
            }
        }
        return byteBuffer;
    }

}
