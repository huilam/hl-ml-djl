package hl.ml.djl.transformer.sbert;

import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;
import hl.ml.djl.DjlModelLoader;

public class SBERT {
	
	private static SBERT instant = null;
	
	private String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private String model_name = "all-MiniLM-L12-v2";
    
	// /src/main/java/hl/ml/djl/transformer/sbert/resources/
	private Predictor<String, float[]> predictor = null;
	
	private SBERT()
	{
		URL url = SBERT.class.getProtectionDomain().getCodeSource().getLocation();
		
		String sResFolder = url.toString()+SBERT.class.getPackageName().replace(".","/")+"/resources/";
		String sModelPath = sResFolder+this.model_name;
		
		//System.out.println("sModelPath="+sModelPath);
		
		String sRtEngine = this.rt_engines[0];
		
		Map<String, Object> mapArgs = new HashMap<>();
	    // Explicitly configure the translator to provide what the model wants
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("includeTokenTypes", sRtEngine.equalsIgnoreCase("OnnxRuntime")?"true":"false"); // This fixes the 'token_type_ids' mismatch
	    
		this.predictor= DjlModelLoader.loadModel(this.rt_engines[0], sModelPath, mapArgs);
	}
	
	public static SBERT getInstance()
	{
		if(instant==null)
		{
			instant = new SBERT();
		}
		return instant;
	}
	
    private double cosineSimilarity(float[] v1, float[] v2) {
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            n1 += v1[i] * v1[i];
            n2 += v2[i] * v2[i];
        }
        return dot / (Math.sqrt(n1) * Math.sqrt(n2));
    }
    
    public double calcSimilarityScore(String aSentence1, String aSentence2) throws TranslateException
    {
		double lSimilarityScore = -1;
    	float[] v1 = predictor.predict(aSentence1);
        float[] v2 = predictor.predict(aSentence2);
        
        lSimilarityScore = cosineSimilarity(v1, v2);
        
        return lSimilarityScore;
    }
}