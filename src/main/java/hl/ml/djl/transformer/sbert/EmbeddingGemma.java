package hl.ml.djl.transformer.sbert;

import java.util.HashMap;
import java.util.Map;

import ai.djl.translate.TranslateException;

public class EmbeddingGemma extends SBERT{
	
	private static EmbeddingGemma instant = null;
	
	private final static String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private final static String model_names[] = new String[]{"embeddinggemma-300m"};
    
	protected EmbeddingGemma()
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