package peersim.multisvm;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

import peersim.config.Configuration;
import peersim.core.Control;
import peersim.core.Network;
import peersim.util.IncrementalStats;

public class Observer implements Control {
	

	

	    private static final String PAR_Threshold = "threshold";

	    private static final String PAR_PROT = "protocol";

	    /** Protocol identifier, obtained from config property {@link #PAR_PROT}. */
	    private final int pid;

	    // private final int prefix;
	    private final String name;

	    private String protocol;

	    private final double threshold;

	    public double avg_f;

	    public double[][] avg_wtVec;

	    public double f;

	    public double[][] w;

	    public int numofclass = MultiSVMCustomNode.write_class_size();

	    public int numofAtt;

	    public Observer(String name)
	    {

	        this.name = name;
	        threshold = 0.0025;

	        this.pid = Configuration.getPid(name + "." + PAR_PROT);
	        // System.out.println(pid);
	        // protocol = Configuration.getString(name + "." + "prot",
	        // "local_model1");
	        // protocol = Configuration.getString(name + "." + "prot",
	        // "local_model1");
	    }

	    // boolean retVal = true;
	    public boolean execute()
	    {

	    	MultiSVMCustomNode n1 = (MultiSVMCustomNode) Network.get(2);
	        numofAtt = n1.num_Att;
	        // TODO Auto-generated method stub
	        if (MultiSvmProtocol.end)
	            return true;
	        IncrementalStats stats = new IncrementalStats();
	        long time = peersim.core.CommonState.getTime();
	        f = 0.0;
	        avg_f = 0.0;
	        avg_wtVec = new double[numofAtt][numofclass];

	        w = new double[numofAtt][numofclass];
	        for (int row = 0; row < numofAtt; row++)
	        {
	            for (int col = 0; col < numofclass; col++)
	            {
	                avg_wtVec[row][col] = 0;
	            }
	        }
	        for (int row = 0; row < numofAtt; row++)
	        {
	            for (int col = 0; col < numofclass; col++)
	            {
	                w[row][col] = 0;
	            }
	        }
	        File file = new File("C:\\Users\\sony\\Desktop\\Thesis\\NewResult\\forestcovertype_result_iter1000000.txt");

	        for (int i = 0; i < Network.size(); i++) {

	            // myNewSVMCode protocol =
	            // (myNewSVMCode)Network.get(i).getProtocol(pid);
	        	MultiSVMCustomNode n = (MultiSVMCustomNode) Network.get(i);
	            // MyNode n1 = (MyNode) n;

	            // for (int row = 0; row < n.num_Att; row++)
	            // {
	            // for (int col = 0; col < n.num_class; col++)
	            // {
	            // System.out.println("The weight matrix at node" + "(" + i + "):" +
	            // n.wtVec[row][col]);
	            // }
	            // }

	           // System.out.println("The frobenius_norm at node" + "(" + i + "):" + n.frobenius_norm);

	            f = f + n.frobenius_norm;
	            for (int row = 0; row < n.num_Att; row++)
	            {
	                for (int col = 0; col < n.num_class; col++)
	                {
	                    w[row][col] = w[row][col] + n.wtVec[row][col];
	                }

	            }
	        }
	        for (int row = 0; row < numofAtt; row++)
	        {
	            for (int col = 0; col < numofclass; col++)
	            {

	                avg_wtVec[row][col] = (w[row][col] / Network.size());
	            }
	        }
//	        for (int row = 0; row < numofAtt; row++)
//	        {
//	            for (int col = 0; col < numofclass; col++)
//	            {
//
//	                System.out.println("The average weight matrix" + ":" + avg_wtVec[row][col]);
//	            }
//	        }
	       // System.out.println("The Average frobenius norm:");
	        avg_f = (f / Network.size());
	        //System.out.println(avg_f);
	        try {
	            FileWriter fw;
	            fw = new FileWriter(file.getAbsoluteFile());
	            BufferedWriter bw = new BufferedWriter(fw);

	            bw.write(String.valueOf(avg_f));
	            bw.write("-----");
//	            for (int row = 0; row < numofAtt; row++)
//	            {
//	                for (int col = 0; col < numofclass; col++)
//	                {
//	                    bw.write(String.valueOf(avg_wtVec[row][col]));
//	                }
//	            }
	            bw.close();
	        }
	        catch (IOException e) {
	            // TODO Auto-generated catch block
	            e.printStackTrace();
	        }

	        //
	        // else
	        // {
	        return false;
	        // }
	    }

	

}
