package hl.ml.djl.transformer.sbert.all_minilm;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;
import hl.ml.djl.CommonConstants;
import hl.ml.djl.transformer.sbert.SBERT;

public class AllMiniLM extends SBERT{
	
	private static AllMiniLM instant = null;
	private final static String model_name		= "all-MiniLM-L12-v2";
    
	protected AllMiniLM()
	{
		Map<String, Object> mapArgs = new HashMap<>();
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("includeTokenTypes", "true");
	    
		super(AllMiniLM.class, CommonConstants.RT_ENGINE_ONNX, model_name, mapArgs);
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