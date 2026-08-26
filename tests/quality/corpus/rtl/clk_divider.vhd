library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity clk_divider is
  -- Frequency divider: divides clk by an integer ratio and gates the
  -- result with enable. The counter counts up to ratio minus one.
  generic (
    RATIO : natural := 2
  );
  port (
    clk     : in  std_logic;
    enable  : in  std_logic;
    clk_div : out std_logic
  );
end entity clk_divider;

architecture rtl of clk_divider is
  signal count : natural range 0 to RATIO - 1 := 0;
  signal out_r : std_logic := '0';
begin
  process (clk, enable)
  begin
    if rising_edge(clk) then
      if enable = '1' then
        if count = RATIO - 1 then
          count <= 0;
          out_r <= not out_r;
        else
          count <= count + 1;
        end if;
      end if;
    end if;
  end process;

  clk_div <= out_r;
end architecture rtl;