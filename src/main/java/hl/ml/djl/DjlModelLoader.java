package hl.ml.djl;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory;
import ai.djl.inference.Predictor;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;

import java.io.File;
import java.io.IOException;

public class DjlModelLoader {
	
	public static Predictor<String, float[]> loadModel(final String aRTEngine, final String aModelPath)
	{
		long lStartMs = System.currentTimeMillis();
		
		File folderModel = new File(aModelPath);
		if(!folderModel.exists())
		{
			System.err.println("folder not exist ! - "+folderModel.getAbsolutePath());
		}
		
		Predictor<String, float[]> predictor = null;
		String sIsIncludeTokenTypes = "false";
		
		if(aRTEngine.equalsIgnoreCase("OnnxRuntime"))
		{
			sIsIncludeTokenTypes = "true";
		}
		
        // In 0.36.0, we use optArgument to pass configuration 
        // and let the ServiceLoader find the translator automatically.
		Criteria<String, float[]> criteria = Criteria.builder()
        	    .setTypes(String.class, float[].class)
        	    .optEngine(aRTEngine)
        	    .optModelUrls(folderModel.getAbsolutePath()) // DJL looks here first
        	    
        	    // FORCE CPU HERE
        	    .optDevice(Device.cpu())
        	   
        	    // Explicitly configure the translator to provide what the model wants
        	    .optArgument("padding", "true")
        	    .optArgument("truncation", "true")
        	    
        	    // true for onnx
        	    .optArgument("includeTokenTypes", sIsIncludeTokenTypes) // This fixes the 'token_type_ids' mismatch
        	    .optTranslatorFactory(new TextEmbeddingTranslatorFactory())
        	    .build();
		
		ZooModel<String, float[]> model = null;
		try {
			model = criteria.loadModel();
		} catch (ModelNotFoundException | MalformedModelException | IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		System.out.println("Model loading time = "+(System.currentTimeMillis()-lStartMs)+" ms");
		
		if(model!=null)
		{
			predictor = model.newPredictor();
		}
		
		return predictor;
	}

}