package hl.ml.djl.transformer.sbert.granite_embedding;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;
import hl.ml.djl.CommonConstants;
import hl.ml.djl.transformer.sbert.SBERT;

public class GraniteEmbedding extends SBERT{
	
	private static GraniteEmbedding instant = null;
	private final static String model_name = "granite-embedding-english-r2";
    
	protected GraniteEmbedding()
	{
		Map<String, Object> mapArgs = new HashMap<>();
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("pooling", "mean"); 
		mapArgs.put("includeTokenTypes", "false"); // Gemma is decoder-only
		
		super(GraniteEmbedding.class, CommonConstants.RT_ENGINE_ONNX, model_name, mapArgs);
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