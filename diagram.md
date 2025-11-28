classDiagram
    %% Base Classes
    class BaseDataClass {
        +validate_inputs()
    }

    class Asset {
        +run_sim()
        +run_sim_arrays()
    }

    %% Data Containers
    class Rates {
        +calc_fwds()
        +interpolate()
    }
    class Liabilities {
        +pv()
    }
    class Issuers {
        +n_sectors
        +n_issuers
    }
    
    %% Inheritance Relationships
    BaseDataClass <|-- Rates
    BaseDataClass <|-- Liabilities
    BaseDataClass <|-- Issuers
    BaseDataClass <|-- Asset
    Asset <|-- Bonds

    %% Core Models
    class TransitionMatrix {
        +transition()
        +transitionv()
    }

    class CreditRiskModel {
        +run()
    }

    class Bonds {
        +pv()
        +sim_pv()
        +sim_cashflows()
    }

    class Portfolio {
        +run_sim()
        +run_sim_arrays()
    }

    %% Mandates
    class CDIMandate {
        +run()
    }

    class CDIMandate_Fox {
        +run()
    }

    CDIMandate <|-- CDIMandate_Fox

    %% Data Classes (Results)
    class SimulationResult
    class CDISimulationResult

    %% Relationships / Composition
    CreditRiskModel --> Issuers
    CreditRiskModel --> TransitionMatrix
    CreditRiskModel ..> SimulationResult : Produces

    Bonds --> Issuers
    
    Portfolio o-- Asset : Contains List of

    CDIMandate --> Liabilities
    CDIMandate --> Portfolio
    CDIMandate ..> CDISimulationResult : Produces