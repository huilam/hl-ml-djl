package hl.ml.djl.transformer.sbert;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;

public class GraniteEmbedding extends SBERT{
	
	private static GraniteEmbedding instant = null;
	
	private final static String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private final static String model_names[] = new String[]{"granite-embedding-english-r2"};
    
	protected GraniteEmbedding()
	{
		String sModelName = model_names[0];
		String sRtEngine = rt_engines[0];
		
		Map<String, Object> mapArgs = new HashMap<>();
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("pooling", "mean"); 
		mapArgs.put("includeTokenTypes", "false"); // Gemma is decoder-only
		
		super(sRtEngine, sModelName, mapArgs);
	}
	
	public static GraniteEmbedding getInstance()
	{
		if(instant==null)
		{
			instant = new GraniteEmbedding();
		}
		return instant;
	}
	
	public static void main(String[] args) throws TranslateException {
		SBERT.unit_test_1( GraniteEmbedding.getInstance() );
    }
}