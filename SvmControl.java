package peersim.multisvm;

import peersim.config.Configuration;
import peersim.core.Control;
import peersim.core.Network;

public class SvmControl implements Control {

		/** 
		 * String name of the parameter used to select the protocol to operate on
		 */
		public static final String PAR_PROTID = "protocol";

		/** The name of this object in the configuration file */
		private final String name;

		/** Protocol identifier */
		private final int pid;

		// iterator counter
		private static int i = 0;
		/**
		 * Creates a new observer and initializes the configuration parameter.
		 */
		public SvmControl(String name) {
			this.name = name;
			this.pid = Configuration.getPid(name + "." + PAR_PROTID);
		  }


		public boolean execute() {
			// TODO Auto-generated method stub
			final int len = Network.size();
			for (int i = 0; i <  len; i++) {
				MultiSVMCustomNode node = (MultiSVMCustomNode) Network.get(i);
			}
			
			
			return false;
		}

	

}
