package hl.ml.djl.transformer.sbert;

import java.net.URISyntaxException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;

public class EmbeddingGemma extends SBERT{
	
	private static EmbeddingGemma instant = null;
	
	private final static String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private final static String model_names[] = new String[]{"embeddinggemma-300m"};
    
	protected EmbeddingGemma()
	{
		
		// Use the classloader to get the actual resource folder
        URL resURL = EmbeddingGemma.class.getResource("resources/" + model_names[0]);
        
        // Convert URL to a URI and then to a clean Path string
        // This removes the "file:" prefix and handles spaces (%20) correctly
        String sModelPath = null;
		try {
			sModelPath = java.nio.file.Paths.get(resURL.toURI()).toAbsolutePath().toString();
		} catch (URISyntaxException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String sRtEngine = rt_engines[0];
		
		Map<String, Object> mapArgs = new HashMap<>();
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("pooling", "mean"); 
		mapArgs.put("includeTokenTypes", "false"); // Gemma is decoder-only
		
		
		super(sRtEngine, sModelPath, mapArgs);
	}
	
	public static EmbeddingGemma getInstance()
	{
		if(instant==null)
		{
			instant = new EmbeddingGemma();
		}
		return instant;
	}
	
	public static void main(String[] args) throws TranslateException {
		
		long lAppStart = System.currentTimeMillis();
		
        String s1 = "The weather is very sunny today.";
        String s2 = "It is a bright and sun-filled day.";
        
        EmbeddingGemma instance = EmbeddingGemma.getInstance();
        
        long lInferenceStart = System.currentTimeMillis();
        System.out.println("Similarity Score: " + instance.calcSimilarityScore(s1, s2));
        System.out.println("Inference Time = "+(System.currentTimeMillis()-lInferenceStart)+" ms");
        
        System.out.println("App Elapsed Time = "+(System.currentTimeMillis()-lAppStart)+" ms");
    }
}