package hl.ml.djl.transformer.sbert;

import java.net.URL;
import java.util.HashMap;
import java.util.Map;

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
		mapArgs.put("includeTokenTypes", sRtEngine.equalsIgnoreCase("OnnxRuntime")?"true":"false"); // This fixes the 'token_type_ids' mismatch
	    
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
	
}