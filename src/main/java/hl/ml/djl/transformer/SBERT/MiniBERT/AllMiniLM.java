package hl.ml.djl.transformer.SBERT.MiniBERT;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;
import hl.ml.djl.DJLConstants;
import hl.ml.djl.transformer.SBERT.BaseSBERT;

public class AllMiniLM extends BaseSBERT{
	
	private static AllMiniLM instant = null;
	private final static String model_name		= "all-MiniLM-L12-v2";
    
	protected AllMiniLM()
	{
		Map<String, Object> mapArgs = new HashMap<>();
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("includeTokenTypes", "true");
	    
		super(AllMiniLM.class, DJLConstants.RT_ENGINE_ONNX, model_name, mapArgs);
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
		BaseSBERT.unit_test_1( AllMiniLM.getInstance() );
    }
	
}