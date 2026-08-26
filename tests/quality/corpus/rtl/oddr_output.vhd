library ieee;
use ieee.std_logic_1164.all;

entity oddr_output is
  -- Odd-rate data output register: wraps the ODDR primitive so a
  -- single-cycle input on the system clock produces an odd-rate
  -- (double-rate) data output on the clock.
  port (
    clk  : in  std_logic;
    d0   : in  std_logic;
    d1   : in  std_logic;
    q    : out std_logic
  );
end entity oddr_output;

architecture rtl of oddr_output is
  component oddr
    port (
      clk : in  std_logic;
      d0  : in  std_logic;
      d1  : in  std_logic;
      q   : out std_logic
    );
  end component oddr;
  -- Instantiated below: d0/d1 are the two phases of the data input.
begin
  u_oddr : oddr
    port map (
      clk => clk,
      d0  => d0,
      d1  => d1,
      q   => q
    );
end architecture rtl;