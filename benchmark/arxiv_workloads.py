"""
Arxiv Workloads Benchmark Queries
"""

papers = [
    {
        "title": "Multi-scale competition in the Majorana-Kondo system",
    },
    {
        "title": "Chondrule formation by collisions of planetesimals containing volatiles triggered by Jupiter's formation",
    },
    
    {
        "title": "Resolving the flat-spectrum conundrum: clumpy aerosol distributions in sub-Neptune atmospheres",
    },
    {
        "title": "A Smooth Transition from Giant Planets to Brown Dwarfs from the Radial Occurrence Distribution",
    },
    {
        "title": "The Albedo Problem and Cloud Cover on Hot Jupiters",
    },
    {
        "title": "Resonant structures in exozodiacal clouds created by exo-Earths in the habitable zone of late-type stars",
    },
    {
        "title": "Evidence of 1:1 slope between rocky Super-Earths and their host stars",
    },
    {
        "title": "Kepler-1624b Has No Significant Transit Timing Variations",
    },
    {
        "title":"Numerical Non-Adiabatic Tidal Calculations with GYRE-tides: The WASP-12 Test Case"
    },
    {
        "title":"Disc breaking and parametric instability in warped accretion discs"
    },
    {
        "title":"Hyperspectral Variational Autoencoders for Joint Data Compression and Component Extraction"
    },
    {
        "title":"Constructing Earth Formation History Using Deep Mantle Noble Gas Reservoirs"
    },
    {
        "title":"Dynamical fractalization of ring systems"
    },
    {
        "title":"Zooming into the water snowline: high resolution water observations of the HL Tau disk"
    },
    {
        "title":"Asteroid phase curve modeling with empirical correction for shape and viewing geometry"  
    },
    {
        "title":"Analytical Solutions for Planet-Scattering Small Bodies"
    },
    {
        "title":"Efficiency of Hydrodynamic Atmospheric Escape in Hot Jupiters and Super Earths"  
    },
    {
        "title":"Excitation of Inertial Modes in 3D Simulations of Rotating Convection in Planets and Stars"
    },
    {
        "title":'Thermal equilibrium curves of accretion disks driven by magnetorotational instability'
    },
    {
        "title":"On the uniqueness of the coupled entropy"
    },
    {
        "title":"Fracture and failure of shear-jammed dense suspensions under impact"
    },
    {
        "title":"Analog Physical Systems Can Exhibit Double Descent"
    },
    {
        "title":"Dissipation anomaly in gradient-driven nonequilibrium steady states"
    },
    {
        "title":"PyAPX: Python toolkit for atomic configuration pattern exploration"
    },
    {
        "title":"Strained hyperbolic Dirac fermions: Zero modes, flat bands, and competing orders" 
    },
    {
        "title":"Chiral spin liquid instability of the Kitaev honeycomb model with crystallographic defects" 
    },
    {
        "title":"Scale-Rich Network-Based Metamaterials" 
    },
    {
        "title":"Observation of topological phases without crystalline counterparts" 
    },
    {
        "title":"Inertia-chirality interplay in active Brownian motion: exact dynamics and phase maps"  
    },
    {
        "title":"Atomistic Framework for Glassy Polymer Viscoelasticity Across 20 Frequency Decades" 
    },
    {
        "title":"Non-Abelian operator size distribution in charge-conserving many-body systems"
    },
    {
        "title":"Adiabatic charge transport through non-Bloch bands"
    }
]

ARXIV_WORKLOADS = []
# 3 papers
for i in [0,1,2]:
    batch = [
    {
        "name": "Query 1: Introduction and Contributions",
        "query": (
            f"Summarize the introduction and core contributions of the paper titled "
            f"'{papers[i]['title']}'. Return the response as a short narrative paragraph."
        )
    },
    {
        "name": "Query 2: Methodology and Conclusions",
        "query": (
            "For this same paper, analyze the methodology in detail, highlighting data sources, "
            "experimental setup, and any assumptions that underpin the approach."
        )
    },
    {
        "name": "Query 3: Introduction and Contributions (repeat)",
        "query": (
            "Now summarize the primary conclusions, implications, and recommended future work for this paper. "
            "Call out any surprising insights separately."
        )
    }
    # {
    #     "name": "Query 4: Contrast Second Paper",
    #     "query": (
    #         f"Summarize the main ideas of '{papers[i+1]['title']}' and explicitly contrast them with each of the above papers "
    #         "across goals, methods, and findings "
    #     )
    # },
    # {
    #     "name": "Query 5: Limitations of Third Paper",
    #     "query": (
    #         f"Summarize only the limitations, open questions, or caveats discussed in the paper "
    #         f"'{papers[i+2]['title']}'. Provide the answer as a concise bullet list."
    #     )
    # }
    ]
    ARXIV_WORKLOADS.append(batch)

def get_arxiv_workload():
    return ARXIV_WORKLOADS

        





