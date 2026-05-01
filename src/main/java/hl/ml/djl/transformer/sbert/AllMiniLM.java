package hl.ml.djl.transformer.sbert;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;

public class AllMiniLM extends SBERT{
	
	private static AllMiniLM instant = null;
	
	private final static String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private final static String model_names[] = new String[]{"all-MiniLM-L12-v2"};
    
	protected AllMiniLM()
	{
		String sModelName = model_names[0];
		String sRtEngine = rt_engines[0];
		
		Map<String, Object> mapArgs = new HashMap<>();
	    // Explicitly configure the translator to provide what the model wants
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("includeTokenTypes", "true");
	    
		super(sRtEngine, sModelName, mapArgs);
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
		SBERT.unit_test_1( AllMiniLM.getInstance() );
    }
	
}