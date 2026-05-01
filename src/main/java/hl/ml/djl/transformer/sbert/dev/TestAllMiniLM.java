package hl.ml.djl.transformer.sbert.dev;

import ai.djl.translate.TranslateException;
import hl.ml.djl.transformer.sbert.AllMiniLM;

public class TestAllMiniLM {
	
	public static void main(String[] args) throws TranslateException {
		
		long lAppStart = System.currentTimeMillis();
		
        String s1 = "The weather is very sunny today.";
        String s2 = "It is a bright and sun-filled day.";
        
        AllMiniLM sbert = AllMiniLM.getInstance();
        
        long lInferenceStart = System.currentTimeMillis();
        System.out.println("Similarity Score: " + sbert.calcSimilarityScore(s1, s2));
        System.out.println("Inference Time = "+(System.currentTimeMillis()-lInferenceStart)+" ms");
        
        System.out.println("App Elapsed Time = "+(System.currentTimeMillis()-lAppStart)+" ms");
    }


}