package hl.ml.djl.transformer.sbert;

import java.net.URL;
import java.util.Map;

import ai.djl.inference.Predictor;
import ai.djl.translate.TranslateException;
import hl.ml.djl.DjlModelLoader;

public class SBERT {
	
	protected String model_name = null;
	protected Predictor<String, float[]> predictor = null;
	
	protected SBERT(final String aRtEngine, String aModelName, Map<String, Object> aMapArgs)
	{
		setModel_name(aModelName);
		
		Class aClass = SBERT.class;
		URL url = aClass.getProtectionDomain().getCodeSource().getLocation();

		String sResFolder = url.toString()+aClass.getPackageName().replace(".","/")+"/resources/";

		this.predictor = DjlModelLoader.loadModel(aRtEngine, sResFolder + getModel_name(), aMapArgs);
	}
	
    public String getModel_name() {
		return model_name;
	}

	public void setModel_name(String model_name) {
		this.model_name = model_name;
	}

	protected double cosineSimilarity(float[] v1, float[] v2) {
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
    	float[] v1 = getEmbedding(aSentence1);
        return calcSimilarityScore(v1, aSentence2);
    }
    
    public double calcSimilarityScore(float[] aEmbedding1, String aSentence2) throws TranslateException
    {
		double lSimilarityScore = -1;
        float[] v2 = getEmbedding(aSentence2);
        lSimilarityScore = calcSimilarityScore(aEmbedding1, v2);
        return lSimilarityScore;
    }
    
    public double calcSimilarityScore(float[] aEmbedding1, float[] aEmbedding2) throws TranslateException
    {
        return cosineSimilarity(aEmbedding1, aEmbedding2);
    }
    
    public float[] getEmbedding(String aSentence) throws TranslateException
    {
    	return predictor.predict(aSentence);
    }
    
	protected static void unit_test_1(SBERT sbert) throws TranslateException {
		
		long lAppStart = System.currentTimeMillis();
		
        String s1 = "The weather is very sunny today.";
        String s2 = "It is a bright and sun-filled day.";
        
        long lInferenceStart = System.currentTimeMillis();
        System.out.println("Model Name: " + sbert.getModel_name());
        System.out.println("Similarity Score: " + sbert.calcSimilarityScore(s1, s2));
        System.out.println("Inference Time = "+(System.currentTimeMillis()-lInferenceStart)+" ms");
        
        System.out.println("App Elapsed Time = "+(System.currentTimeMillis()-lAppStart)+" ms");
    }
}