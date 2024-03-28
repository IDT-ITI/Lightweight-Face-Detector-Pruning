package com.example.testkotlin;

import android.content.Context;

import org.pytorch.IValue;
import org.pytorch.LiteModuleLoader;
import org.pytorch.Tensor;
import org.pytorch.Module;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class EResFD {
    // EResFD model class
    private Module model;

    public EResFD(Context context) {
        try {
            String path = EResFD.assetFilePath(context, "lite_scripted_model.ptl");
            System.out.println("Loading model!!!");
            model = LiteModuleLoader.load(path);
            System.out.println("Model loaded!!!");
        } catch (IOException e) {
            System.out.println("Model could not be loaded!");
        }
    }

    public Tensor inference(Tensor inputTensor) {
        // InputTensor is a [h,w,d] tensor
        try {
            IValue inputTensor2 = IValue.from(inputTensor);
            return model.forward(inputTensor2).toTensor();
        } catch (Exception e) {
            System.err.println("Error during inference: " + e.getMessage());
            return null;
        }
    }

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }

}
