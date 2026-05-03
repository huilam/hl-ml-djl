package hl.ml.djl.transformer.sbert.embeddinggemma;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;
import hl.ml.djl.CommonConstants;
import hl.ml.djl.transformer.sbert.SBERT;

public class EmbeddingGemma extends SBERT{
	
	private static EmbeddingGemma instant = null;
	
	private final static String model_name = "embeddinggemma-300m";
    
	protected EmbeddingGemma()
	{	
		Map<String, Object> mapArgs = new HashMap<>();
		mapArgs.put("padding", "true");
		mapArgs.put("truncation", "true");
		mapArgs.put("pooling", "mean"); 
		mapArgs.put("includeTokenTypes", "false"); // Gemma is decoder-only
		
		super(EmbeddingGemma.class, CommonConstants.RT_ENGINE_ONNX, model_name, mapArgs);
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
		SBERT.unit_test_1( EmbeddingGemma.getInstance() );
    }
}