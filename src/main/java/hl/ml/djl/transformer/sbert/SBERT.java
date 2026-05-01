package hl.ml.djl.transformer.sbert;

import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;
import hl.ml.djl.DjlModelLoader;

public class SBERT {
	
	private static SBERT instant = null;
	
	private String rt_engines[]  = new String[]{"OnnxRuntime","PyTorch"};
	private String model_path = "./src/main/java/hl/ml/djl/transformer/sbert/resources/all-MiniLM-L12-v2/";
    
	// /src/main/java/hl/ml/djl/transformer/sbert/resources/
	private Predictor<String, float[]> predictor = null;
	
	private SBERT()
	{
		this.predictor= DjlModelLoader.loadModel(this.rt_engines[0], this.model_path);
	}
	
	public static SBERT getInstance()
	{
		if(instant==null)
		{
			instant = new SBERT();
		}
		return instant;
	}
	
    private double cosineSimilarity(float[] v1, float[] v2) {
        double dot = 0, n1 = 0, n2 = 0;
        for (int i = 0; i < v1.length; i++) {
            dot += v1[i] * v2[i];
            n1 += v1[i] * v1[i];
            n2 += v2[i] * v2[i];
        }
        return dot / (Math.sqrt(n1) * Math.sqrt(n2));
    }
    
    public double calcSimilarityScore(String aSentence1, String aSentence2) throws TranslateException
    {
		double lSimilarityScore = -1;
    	float[] v1 = predictor.predict(aSentence1);
        float[] v2 = predictor.predict(aSentence2);
        
        lSimilarityScore = cosineSimilarity(v1, v2);
        
        return lSimilarityScore;
    }
}