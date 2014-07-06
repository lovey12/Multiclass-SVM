package peersim.MultiSvm;
import java.util.Iterator;
import java.util.TreeMap;

import peersim.cdsim.CDProtocol;
import peersim.config.*;
import peersim.core.*;
import peersim.vector.*;

public class myNewSVMObserver implements Control{
	private static final String PAR_Threshold = "threshold";
	private static final String PAR_PROT = "protocol";
	/** Protocol identifier, obtained from config property {@link #PAR_PROT}. */
	private final int pid;
	//private final int prefix;
	private final String name;
	private String protocol;
	private final double threshold;
	

	public myNewSVMObserver(String name){
		this.name=name;
    threshold= 0.002;
	this.pid = Configuration.getPid(name + "." + PAR_PROT);
	//System.out.println(pid);
	//protocol = Configuration.getString(name + "." + "prot", "local_model1");
	//protocol = Configuration.getString(name + "." + "prot", "local_model1");
	}
	boolean retVal = true;

	public boolean execute() {
		// TODO Auto-generated method stub
		//if(myNewSVMCode.end) return true;
		
		for (int i = 0; i <Network.size(); i++) {
			
			MyNode n = (MyNode)Network.get(i);
			//MyNode n1 = (MyNode) n;		
           
           System.out.println("The frobenius_norm at node"+"("+i+"):"+n.frobenius_norm);
			retVal=retVal &&(n.frobenius_norm<=threshold);
			System.out.println(retVal);
         }
       if(retVal)
    {
    	   System.out.println("ALgorithm Converged...###########################!");
    	   return myNewSVMCode.end;
    }
       else	
       {
		return false;
       }
       }

}
