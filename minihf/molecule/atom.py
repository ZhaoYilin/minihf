import numpy as np
import os

class Element:
    """Element class containing atomic properties"""
    
    # Class variable to store periodic table data
    _periodic_table = {}
    
    def __init__(self, symbol=None, number=None):
        """
        Initialize an element by symbol or atomic number
        
        Args:
            symbol: Element symbol (e.g., 'H', 'He')
            number: Atomic number (e.g., 1, 2)
        """
        # Load periodic table data if not already loaded
        self._load_periodic_table()
        
        if symbol is not None:
            self._init_by_symbol(symbol)
        elif number is not None:
            self._init_by_number(number)
        else:
            raise ValueError("Either symbol or number must be provided")

    @classmethod
    def _load_periodic_table(cls):
        """Load periodic table data from CSV file"""
        if cls._periodic_table:
            return
        
        data_file = os.path.join(os.path.dirname(__file__), './', 'periodic_table.csv')
         
        # Load the CSV file
        data_all = np.loadtxt(data_file, delimiter=',', dtype=str)
        data_headers = data_all[0]  # Get headers
        data_type = data_all[1]  # Get data rows
        data = data_all[2:]
        
        header_to_index = {header: i for i, header in enumerate(data_headers)}
        
        for row in data:
            element_data = {}
            
            # Automatically process each column based on header and data type
            for header, dtype in zip(data_headers, data_type):
                value = row[header_to_index[header]]
                
               # Skip empty values
                if value == '' or value is None:
                    element_data[str(header)] = None
                    continue
                
                # Convert based on data type
                try:
                    value = eval(dtype)(value)   
                    element_data[str(header)] = value
                except ValueError:
                    # If conversion fails, keep as string
                    element_data[str(header)] = value                

            # Store by both symbol and atomic number
            cls._periodic_table[element_data['symbol']] = element_data
            cls._periodic_table[element_data['number']] = element_data        
            
    def _init_by_symbol(self, symbol):
        """Initialize by element symbol"""
        if symbol not in self._periodic_table:
            raise ValueError(f"Element '{symbol}' not found in periodic table")
        
        data = self._periodic_table[symbol]
        for key, value in data.items():
            setattr(self, key, value)
         
    def _init_by_number(self, number):
        """Initialize by atomic number"""
        if number not in self._periodic_table:
            raise ValueError(f"Atomic number {number} not found in periodic table")
        
        data = self._periodic_table[number]
        for key, value in data.items():
            setattr(self, key, value)

    
class Atom(object):
    
    def __init__(self, element, coords):
        """
        Initialize an atom
        
        Parameters
        ----------
        element : Element or str or int
            Can be an Element object, element symbol (str), or atomic number (int)
        coords : list
            [x, y, z] coordinates of the atom
        """
        # Handle different input types for element
        if isinstance(element, Element):
            # Already an Element object
            for name, value in element.__dict__.items():
                setattr(self, name, value)
        elif isinstance(element, str):
            # Element symbol
            elem = Element(symbol=element)
            for name, value in elem.__dict__.items():
                setattr(self, name, value)
        elif isinstance(element, int):
            # Atomic number
            elem = Element(number=element)
            for name, value in elem.__dict__.items():
                setattr(self, name, value)
        else:
            raise TypeError(f"element must be Element, str, or int, not {type(element)}")
        
        self.assign_coords(coords)

    def assign_coords(self, coords):
        """
        Assign coordinates to the atom
        
        Parameters
        ----------
        coords : list
            [x, y, z] coordinates
        
        Raises
        ------
        TypeError
            If coords is not a list
        ValueError
            If coords length is not 3
        """
        if not isinstance(coords, list):
            raise TypeError("coords must be list")
        if not len(coords) == 3:
            raise ValueError("length of the coords must be 3")
        self.coords = np.array(coords, dtype=float)

    def __eq__(self, other, error=1e-6):
        """Check if two atoms are equal"""
        if not isinstance(other, Atom):
            return False
        if self.number != other.number:
            return False
        if not np.allclose(self.coords, other.coords, error):
            return False
        return True