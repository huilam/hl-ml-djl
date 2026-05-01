package hl.ml.djl.transformer.sbert;

import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;

public class AllMiniLM extends SBERT{
	
	private static AllMiniLM instant = null;
	
	private final static String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private final static String model_names[] = new String[]{"all-MiniLM-L12-v2"};
    
	protected AllMiniLM()
	{
		URL url = AllMiniLM.class.getProtectionDomain().getCodeSource().getLocation();
		
		String sResFolder = url.toString()+AllMiniLM.class.getPackageName().replace(".","/")+"/resources/";
		String sModelPath = sResFolder+model_names[0];
		String sRtEngine = rt_engines[0];
		
		Map<String, Object> mapArgs = new HashMap<>();
	    // Explicitly configure the translator to provide what the model wants
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("includeTokenTypes", "true");
	    
		super(sRtEngine, sModelPath, mapArgs);
	}
	
	public static AllMiniLM getInstance()
	{
		if(instant==null)
		{
			instant = new AllMiniLM();
		}
		return instant;
	}
	
	
	
	public static void main(String[] args) throws TranslateException {
		
		long lAppStart = System.currentTimeMillis();
		
        String s1 = "The weather is very sunny today.";
        String s2 = "It is a bright and sun-filled day.";
        
        AllMiniLM sbert = AllMiniLM.getInstance();
        
        long lInferenceStart = System.currentTimeMillis();
        System.out.println("Similarity Score: " + sbert.calcSimilarityScore(s1, s2));
        System.out.println("Inference Time = "+(System.currentTimeMillis()-lInferenceStart)+" ms");
        
        System.out.println("App Elapsed Time = "+(System.currentTimeMillis()-lAppStart)+" ms");
    }
	
}